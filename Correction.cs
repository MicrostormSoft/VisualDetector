using OpenCvSharp;
using System.Linq;
using System.Collections.Generic;

namespace VisualDetector
{
    public static class Correction
    {
        public struct KeyPoints
        {
            public Point2f TopLeft, TopRight, BottomRight, BottomLeft;
        }

        public static Mat ImageRectification(Mat image, KeyPoints imagepoints, KeyPoints targetpoints, Size OutputSize)
        {
            Point2f[] pts_src = {
                imagepoints.TopLeft,imagepoints.TopRight,imagepoints.BottomLeft,imagepoints.BottomRight
            };
            Point2f[] pts_dst = {
                targetpoints.TopLeft,targetpoints.TopRight,targetpoints.BottomLeft,targetpoints.BottomRight
            };

            Mat M = Cv2.GetPerspectiveTransform(pts_src, pts_dst);
            return image.WarpPerspective(M, OutputSize);
        }

        public static Mat Auto9PBoardCorrection(Mat mat, int mp = 8, int sized = 30, int graythreshold = 140)
        {
            var binframe = mat.CvtColor(ColorConversionCodes.BGR2GRAY).Blur(new Size(sized, sized));

            Cv2.Threshold(binframe, binframe, graythreshold, 255, ThresholdTypes.Binary);
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Close, kernel, new Point(-1, -1));
            kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Open, kernel, new Point(-1, -1));
            Cv2.FindContours(binframe, out Point[][] conts, out HierarchyIndex[] h, RetrievalModes.External, ContourApproximationModes.ApproxNone);
            //分离不连续的散点

            List<Point> RectPoints = new List<Point>();

            foreach (var shape in conts)
            {
                if (!Detection.Circle(shape, out Point center, out double confidence, out double R)) continue;//是圆?
                RectPoints.Add(center);
            }
            if (RectPoints.Count >= 9)//有超过9个标记点
            {
                var keypoints = GetKeypoints(RectPoints);
                return Correction.ImageRectification(mat, keypoints,
                      new Correction.KeyPoints
                      {
                          TopLeft = new Point2f(12.5f * mp, 12.5f * mp),
                          TopRight = new Point2f(52.5f * mp, 12.5f * mp),
                          BottomLeft = new Point2f(12.5f * mp, 52.5f * mp),
                          BottomRight = new Point2f(52.5f * mp, 52.5f * mp)
                      }
                      , new Size(65 * mp, 65 * mp));
            }
            else
            {//少于9个标记点，无法操作
                return mat;
            }
        }

        public static Correction.KeyPoints GetKeypoints(IEnumerable<Point> points)
        {
            Point zero = new Point(0, 0);
            Correction.KeyPoints result = new Correction.KeyPoints
            {
                TopLeft = points.First(),
                BottomRight = points.First(),
                TopRight = points.First(),
                BottomLeft = points.First()
            };
            foreach (Point p in points)
            {
                if (result.TopLeft.DistanceTo(zero) > p.DistanceTo(zero)) result.TopLeft = p;
                if (result.BottomRight.DistanceTo(zero) < p.DistanceTo(zero)) result.BottomRight = p;
            }
            Point TopRightConner = new Point(result.BottomRight.X, result.TopLeft.Y);
            foreach (Point p in points)
            {
                if (p == result.TopLeft || p == result.BottomRight) continue;
                if (result.TopRight.DistanceTo(TopRightConner) > p.DistanceTo(TopRightConner)) result.TopRight = p;
                if (result.BottomLeft.DistanceTo(TopRightConner) < p.DistanceTo(TopRightConner)) result.BottomLeft = p;
            }
            return result;
        }
    }
}
