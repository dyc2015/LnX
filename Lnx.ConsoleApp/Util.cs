using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lnx.LeeCode
{
    internal class Util
    {
        /// <summary>
        /// 生成随机数组
        /// </summary>
        /// <param name="count">数组个数</param>
        /// <param name="max">最大值</param>
        /// <returns></returns>
        public static int[] GenerateIntArray(int count, int max)
        {
            int[] array = new int[count];
            Random rnd = new();
            for (int i = 0; i < count; i++)
            {
                 array[i] = rnd.Next(-max, max);
            }

            return array;
        }
    }
}
