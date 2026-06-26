#!/usr/bin/env python3
import argparse
import sys

import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image


def compressed_to_image(msg, encoding):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")

    out = Image()
    out.header = msg.header
    out.height, out.width = image.shape[:2]
    out.encoding = encoding
    out.is_bigendian = False
    out.step = image.strides[0]
    out.data = image.tobytes()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Replace one CompressedImage topic in a rosbag with a raw sensor_msgs/Image topic."
    )
    parser.add_argument("input_bag")
    parser.add_argument("output_bag")
    parser.add_argument("--compressed-topic", default="/camera/color/image_raw/compressed")
    parser.add_argument("--raw-topic", default="/camera/color/image_raw")
    parser.add_argument("--encoding", default="bgr8")
    parser.add_argument("--keep-compressed", action="store_true")
    args = parser.parse_args()

    converted = 0
    failed = 0
    total = 0

    with rosbag.Bag(args.input_bag, "r") as in_bag, rosbag.Bag(args.output_bag, "w") as out_bag:
        for topic, msg, t in in_bag.read_messages():
            total += 1
            if topic == args.compressed_topic:
                try:
                    raw = compressed_to_image(msg, args.encoding)
                except Exception as exc:
                    failed += 1
                    print(f"[decompress_bag] failed at {msg.header.stamp.to_sec():.6f}: {exc}", file=sys.stderr)
                    if args.keep_compressed:
                        out_bag.write(topic, msg, t)
                    continue

                if args.keep_compressed:
                    out_bag.write(topic, msg, t)
                out_bag.write(args.raw_topic, raw, t)
                converted += 1
            else:
                out_bag.write(topic, msg, t)

            if total % 10000 == 0:
                print(f"[decompress_bag] processed={total} converted={converted} failed={failed}")

    print(f"[decompress_bag] done processed={total} converted={converted} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
