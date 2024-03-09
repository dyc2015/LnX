using LnX.ML.CNN;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML
{
    public class MinstReader
    {
        public static MinstData[] Read(Stream data, Stream label)
        {
            var dataCount = label.Length - 8;
            var result = new MinstData[dataCount];

            data.Seek(16, SeekOrigin.Begin);
            label.Seek(8, SeekOrigin.Begin);

            for (int i = 0; i < dataCount; i++)
            {
                var pixels = new Tensor(28, 28);
                for (int j = 0; j < 28; j++)
                {
                    for (var k = 0; k < 28; k++)
                    {
                        pixels[0, 0, j, k] = data.ReadByte() / 255d;
                    }
                }

                result[i] = new MinstData(pixels, ToLabels(label.ReadByte()));
            }

            return result;
        }

        public static double[] ToLabels(int label)
        {
            return label switch
            {
                0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                4 => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                6 => [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                7 => [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                8 => [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                _ => throw new Exception("参数错误"),
            };
        }
    }
}
