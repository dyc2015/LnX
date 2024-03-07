using LnX.ML.CNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace LnX.ML
{
    public struct MinstData(ITensor pixels, double[] labels)
    {
        public ITensor Pixels = pixels;
        public double[] Labels = labels;
    }
}
