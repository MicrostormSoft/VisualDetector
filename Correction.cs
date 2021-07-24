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

        /// <summary>
        /// Rectificate image
        /// </summary>
        /// <param name="image">just the image</param>
        /// <param name="imagepoints">4 points on the image</param>
        /// <param name="targetpoints">where these 4 points should be at</param>
        /// <param name="OutputSize">How big should the output mat be. Output will be resized to this size.</param>
        /// <returns>Output image</returns>
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

        /// <summary>
        /// Correction for a 9-point-board which is 65*65.
        /// <para>The board has 9 while circles on a black background, the center of the conner circles are 12.5 away from both edges.</para>
        /// </summary>
        /// <param name="mat">image to process</param>
        /// <param name="mp">resize rate for getting the proper mat size</param>
        /// <param name="sized">diameter of the white circle. for blur.</param>
        /// <param name="graythreshold">threshold use for graying the image</param>
        /// <returns>corrected image</returns>
        public static Mat Auto9PBoardCorrection(Mat mat, int mp = 8, int sized = 30, int graythreshold = 140, double centerth = 0.2, double singleth = 0.2)
        {
            var binframe = mat.CvtColor(ColorConversionCodes.BGR2GRAY).Blur(new Size(sized, sized));

            Cv2.Threshold(binframe, binframe, graythreshold, 255, ThresholdTypes.Binary);
            //Cv2.AdaptiveThreshold(binframe, binframe, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, graythreshold, 0);
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Close, kernel, new Point(-1, -1));
            kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Open, kernel, new Point(-1, -1));
            Cv2.FindContours(binframe, out Point[][] conts, out HierarchyIndex[] h, RetrievalModes.External, ContourApproximationModes.ApproxNone);
            //Cv2.DrawContours(mat, conts, -1, Scalar.Green);
            //分离不连续的散点

            List<Point> RectPoints = new List<Point>();

            foreach (var shape in conts)
            {
                if (!Detection.Circle(shape, out Point center, out double confidence, out double R, centerth, singleth)) continue;//是圆?
                RectPoints.Add(center);
                //mat.DrawMarker(center, Scalar.Blue, markerSize: 30);
            }
            if (RectPoints.Count >= 4)//有超过4个标记点
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

        /// <summary>
        /// Get the four conner point from lots of points.
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
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
