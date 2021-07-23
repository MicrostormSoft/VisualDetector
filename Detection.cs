using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VisualDetector
{
    public static class Detection
    {
        public static List<Point> DetectRedBall(Mat colorimage,int balld = 8)
        {
            //var binframe = R2Gary(colorimage * 0.5).Blur(new Size(10, 10));
            var binframe = colorimage.InRange(new Scalar(0, 0, 178), new Scalar(148, 150, 255)).Blur(new Size(balld, balld));
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Close, kernel, new Point(-1, -1));
            kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3), new Point(-1, -1));
            Cv2.MorphologyEx(binframe, binframe, MorphTypes.Open, kernel, new Point(-1, -1));

            Cv2.FindContours(binframe, out Point[][] conts, out HierarchyIndex[] h, RetrievalModes.External, ContourApproximationModes.ApproxNone);//分离不连续的散点,分离物体
            colorimage.DrawContours(conts, -1, Scalar.Green);
            List<Point> RectPoints = new List<Point>();

            foreach (var shape in conts)
            {
                Rect r = Cv2.BoundingRect(shape);
                Point center = (r.BottomRight + r.TopLeft);
                center.X /= 2;
                center.Y /= 2;
                RectPoints.Add(center);
            }
            return RectPoints;
        }

        public static bool Circle(Point[] pg, out Point center, out double R, out double confidence, double centerth = 0.1, double singleth = 0.2)
        {
            Point Left = pg[0], Up = pg[0], Right = pg[0], Down = pg[0];
            foreach (var p in pg)
            {
                if (p.X < Left.X) Left = p;
                if (p.X > Right.X) Right = p;
                if (p.Y < Down.Y) Down = p;
                if (p.Y > Up.Y) Up = p;
            }
            center = new Point((Left.X + Right.X) / 2, (Up.Y + Down.Y) / 2);
            R = ((Right.X - Left.X) + (Up.Y - Down.Y)) / 4;//平均半径
            int miss = 0;
            double missrate = 0;
            foreach (var p in pg)
            {
                var d2d = p - center;
                double distance = Math.Sqrt((d2d.X * d2d.X) + (d2d.Y * d2d.Y));
                if (Math.Abs(distance - R) > R * centerth) miss++;
                missrate += Math.Abs(distance - R);
            }
            missrate /= R * pg.Length;
            confidence = 1 - missrate;
            return (miss < (pg.Length * singleth));
        }

        public static Mat R2Gary(Mat input)
        {
            var output = Cv2.Split(input);
            var red = output[2];
            var green = output[1];
            var blue = output[0];
            var reducedred = red - green - blue;
            return reducedred;
        }
    }
}
