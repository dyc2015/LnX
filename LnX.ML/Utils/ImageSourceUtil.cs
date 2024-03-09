using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using LnX.ML.CNN;

namespace LnX.ML.Utils
{
    public class ImageSourceUtil
    {
        /// <summary>
        /// 创建图片源
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static BitmapSource[] CreateBitmapSource(ITensor tensor)
        {
            var w = tensor.Width;
            var h = tensor.Height;
            var bytes = new byte[w * h];
            var pf = PixelFormats.Gray8;

            var result = new BitmapSource[tensor.Num];
            for (int n = 0; n < tensor.Num; n++)
            {
                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        bytes[i * w + j] = (byte)(tensor[n, 0, i, j] * 255);
                    }
                }

                result[n] = BitmapSource.Create(w, h, 96, 96, pf, null, bytes, (w * pf.BitsPerPixel + 7) / 8);
            }

            return result;
        }
    }
}
