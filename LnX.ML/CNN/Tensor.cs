using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML.CNN
{
    /// <summary>
    /// 张量
    /// </summary>
    public interface ITensor
    {
        /// <summary>
        /// 获取或设置某张量
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        double[][,] this[int n] { get; set; }
        /// <summary>
        /// 获取或设置某维度平面
        /// </summary>
        /// <param name="n"></param>
        /// <param name="d"></param>
        /// <returns></returns>
        double[,] this[int n, int d] { get; set; }
        /// <summary>
        /// 获取或设置某值
        /// </summary>
        /// <param name="n"></param>
        /// <param name="d"></param>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <returns></returns>
        double this[int n, int d, int w, int h] { get; set; }
        /// <summary>
        /// 宽度
        /// </summary>
        int Width { get; }
        /// <summary>
        /// 长度
        /// </summary>
        int Height { get; }
        /// <summary>
        /// 维度
        /// </summary>
        int Dimension { get; }
        /// <summary>
        /// 数量
        /// </summary>
        int Num { get; }

        /// <summary>
        /// 数据
        /// </summary>
        double[][][,] Datas { get; }

        ///// <summary>
        ///// 获取范围数据
        ///// </summary>
        ///// <param name="width">宽</param>
        ///// <param name="height">高</param>
        ///// <param name="dimension">维度</param>
        ///// <param name="num">数量</param>
        ///// <returns></returns>
        //double[,] GetDataRange(int width, int height, int num = 0, int dimension = 0);
    }

    /// <summary>
    /// 张量
    /// </summary>
    public class Tensor : ITensor
    {
        public int Width { get; }
        public int Height { get; }
        public int Dimension { get; }
        public int Num { get; }
        public double[][][,] Datas { get; }

        public double this[int n, int d, int w, int h] { get => Datas[n][d][w, h]; set => Datas[n][d][w, h] = value; }
        public double[,] this[int n, int d] { get => Datas[n][d]; set => Datas[n][d] = value; }
        public double[][,] this[int n] { get => Datas[n]; set => Datas[n] = value; }

        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="dimension"></param>
        /// <param name="num"></param>
        public Tensor(int width, int height, int num = 1, int dimension = 1)
        {
            Width = width;
            Height = height;
            Dimension = dimension;
            Num = num;

            Datas = new double[num][][,];
            for (int i = 0; i < num; i++)
            {
                Datas[i] = new double[dimension][,];
                for (int j = 0; j < dimension; j++)
                {
                    Datas[i][j] = new double[width, height];
                }
            }
        }

        //public double[,] GetPlane(int num = 0, int dimension = 0, int width = 0, int height = 0)
        //{
        //    var result = Datas[num][dimension];
        //    if (width > 0 || height > 0)
        //    {
        //        var tmp = new double[width, height];
        //        for (int i = 0; i < width; i++)
        //        {
        //            for (int j = 0; j < height; j++)
        //            {
        //                tmp[i, j] = result[i, j];
        //            }
        //        }

        //        return tmp;
        //    }

        //    return result;
        //}
    }
}
