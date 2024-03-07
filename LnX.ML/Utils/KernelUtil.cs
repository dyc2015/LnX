using LnX.ML.CNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML.Utils
{
    public class KernelUtil
    {
        /// <summary>
        /// 创建卷积核并随机初始化
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="dimension"></param>
        /// <param name="num"></param>
        /// <returns></returns>
        public static Tensor Create(int width, int height, int num = 1, int dimension = 1)
        {
            var result = new Tensor(width, height, num, dimension);
            var random = new Random();

            for (int i = 0; i < num; i++)
                for (int j = 0; j < dimension; j++)
                    for (int i1 = 0; i1 < width; i1++)
                        for (var j1 = 0; j1 < height; j1++)
                            result[i, j, i1, j1] = random.NextDouble();

            return result;
        }
    }
}
