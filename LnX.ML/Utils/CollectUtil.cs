using LnX.ML.CNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML.Utils
{
    public static class CollectUtil
    {
        /// <summary>
        /// 循环所有元素（一行行读）
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="action"></param>
        public static void ForEach<T>(this T[,] source, Action<T> action)
        {
            if (source == null) return;

            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (var j = 0; j < source.GetLength(1); j++)
                {
                    action(source[i, j]);
                }
            }
        }

        /// <summary>
        /// 展平张量
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static double[] Flatten(this ITensor tensor)
        {
            var result = new double[tensor.Num * tensor.Dimension * tensor.Width * tensor.Height];
            for (int i = 0; i < tensor.Num; i++)
            {
                for (var j = 0; j < tensor.Dimension; j++)
                {
                    for (int k = 0; k < tensor.Height; k++)
                    {
                        for (int l = 0; l < tensor.Width; l++)
                        {
                            result[i * tensor.Dimension * tensor.Height * tensor.Width + j * tensor.Height * tensor.Width + k * tensor.Width + l] = tensor[i, j, l, k];
                        }
                    }
                }
            }

            return result;
        }
    }
}
