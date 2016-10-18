# Import needed modules
import video2gif
import optparse
import json
from moviepy.editor import VideoFileClip

def parser():
    parser = optparse.OptionParser()
    parser.add_option("-s", "--source", default="./videos/FrG4TEcSuRg.mp4", help="Which video to process")
    parser.add_option("-d", "--duration", default=3, help="Duration of the segments", type="int")
    parser.add_option("-t", "--top", default=5, help="How many top segments to get", type="int")
    parser.add_option("-b", "--bottom", default=0, help="How many bottom segments to get", type="int")
    return parser.parse_args()

def main():
    args, opts = parser()
    scored_segments = get_scored_segments(args.video, args.duration, args.top, args.bottom)
    print(json.dumps(scored_segments, indent=4))

if __name__ == "__main__":
    main()
