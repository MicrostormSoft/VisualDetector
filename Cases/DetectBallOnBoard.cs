using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VisualDetector;

namespace VisualDetector.Cases
{
    public static class DetectBallOnBoard
    {
        public static Point[] GetLocations(Mat image,int sizer=6)
        {
            var flat = Correction.Auto9PBoardCorrection(image,mp: sizer);
            var locs = Detection.DetectRedBall(flat).ToArray();
            for(int i=0;i<locs.Length;i++)
            {
                locs[i].X /= sizer;
                locs[i].Y /= sizer;
            }
            return locs;
        }
    }
}
