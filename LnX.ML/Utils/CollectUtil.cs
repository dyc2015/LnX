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

            source.ForEach((i, j, x) => action(x));
        }

        /// <summary>
        /// 循环所有元素（一行行读）
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="action"></param>
        public static void ForEach<T>(this T[,] source, Action<int, int ,T> action)
        {
            if (source == null) return;

            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (var j = 0; j < source.GetLength(1); j++)
                {
                    action(i, j, source[i, j]);
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

        /// <summary>
        /// 展平张量
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static ITensor FillNew(this IEnumerable<double> array, int width, int height, int num, int dimension)
        {
            var result = new Tensor(width, height, num, dimension);

            FillTo(array, result);

            return result;
        }

        /// <summary>
        /// 展平张量
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static void FillTo(this IEnumerable<double> array, ITensor tensor)
        {
            int dimension = tensor.Dimension,
                width = tensor.Width,
                height = tensor.Height,
                dc = dimension * width * height,
                hc = width * height,
                w = 0, h = 0, n = 0, d = 0;
            for (int i = 0; i < array.Count(); i++)
            {
                var t = i;
                if (t / dc > 0)
                {
                    t %= dc;
                    if (t == 0)
                    {
                        d = 0; w = 0; h = 0; n++;
                    }
                }

                if (t / hc > 0)
                {
                    t %= hc;
                    if (t == 0)
                    {
                       w = 0; h = 0; d++;
                    }
                }

                if (t / width > 0 && t % width == 0)
                {
                    h = 0; w++;
                }

                tensor[n, d, w, h] = array.ElementAt(i);

                h++;
            }
        }
    }
}
