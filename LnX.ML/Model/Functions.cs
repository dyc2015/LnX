using LnX.ML.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML
{
    public interface IFunction<T>
    {
        /// <summary>
        /// 计算
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        double Compute(T input);
    }

    public interface IDifferentiableFunction<T1> : IFunction<T1>
    {
        /// <summary>
        /// 求导
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        double Differentiate(T1 input);
    }

    public interface IDifferentiableFunction<T1, T2> : IFunction<T1>
    {
        /// <summary>
        /// 求导
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        double Differentiate(T2 input);
    }

    public class Function
    {
        public static ReLUFunction CreateReLU() => new();
        public static MaxPoolingFunction CreateMaxPooling() => new();
        public static AvgPoolingFunction CreateAvgPooling() => new();
        public static SoftmaxFunction CreateSoftMax() => new();
        public static NormalFunction CreateNormal() => new();
        public static MeanSquaredLossFunction CreateMSE() => new();
        public static CrossEntropyLossFunction CreateNCrossEntropy() => new();
    }

    /// <summary>
    /// ReLU函数
    /// </summary>
    public class ReLUFunction : IDifferentiableFunction<double>
    {
        public double Compute(double input)
        {
            if (input < 0) return 0;
            return input;
        }

        public double Differentiate(double input)
        {
            if (input < 0) return 0;
            return 1;
        }
    }

    /// <summary>
    /// 最大值池化函数
    /// </summary>
    public class MaxPoolingFunction : IFunction<double[,]>
    {
        public double Compute(double[,] input)
        {
            var result = 0d;

            input.ForEach(x =>
            {
                if (x > result) result = x;
            });

            return result;
        }
    }


    /// <summary>
    /// 平均值值池化函数
    /// </summary>
    public class AvgPoolingFunction : IFunction<double[,]>
    {

        public double Compute(double[,] input)
        {
            var result = 0d;

            input.ForEach(x =>
            {
                result += x;
            });

            return result / input.Length;
        }
    }

    /// <summary>
    /// softmax
    /// </summary>
    public class SoftmaxFunction : IDifferentiableFunction<(int, double[]), (int, double[], double[])>
    {
        public double Compute((int, double[]) input)
        {
            return Math.Exp(input.Item2[input.Item1]) / input.Item2.Sum(Math.Exp);
        }

        public double Differentiate((int, double[], double[]) input)
        {
            var result = 0d;
            var current = input.Item2[input.Item1];
            for (var i = 0; i < input.Item2.Length; i++)
            {
                if (i == input.Item1)
                {
                    result += input.Item3[i] * (1 - current) * current;
                }
                else
                {
                    result += input.Item3[i] * -current * input.Item2[i];
                }
            }

            return result;
        }
    }

    /// <summary>
    /// y = x
    /// </summary>
    public class NormalFunction : IDifferentiableFunction<double>
    {
        public double Compute(double input)
        {
            return input;
        }

        public double Differentiate(double input)
        {
            return 1;
        }
    }

    /// <summary>
    /// 误差函数
    /// </summary>
    public interface ILossFunction : IDifferentiableFunction<(double[], double[]), (int, double[], double[])>
    {

    }

    /// <summary>
    /// 均方误差函数
    /// </summary>
    public class MeanSquaredLossFunction : ILossFunction
    {
        public double Compute((double[], double[]) input)
        {
            var len = input.Item1.Length;
            if (len != input.Item2.Length) throw new Exception("参数错误");

            var sum = 0d;
            for (int i = 0; i < len; i++)
            {
                sum += Math.Pow(input.Item2[i] - input.Item1[i], 2);
            }

            return sum / len / 2d;
        }

        public double Differentiate((int, double[], double[]) input)
        {
            return (input.Item3[input.Item1] - input.Item2[input.Item1]) / input.Item3.Length;
        }
    }

    /// <summary>
    /// 交叉熵
    /// </summary>
    public class CrossEntropyLossFunction : ILossFunction
    {
        public double Compute((double[], double[]) input)
        {
            if (input.Item1.Length != input.Item2.Length) throw new Exception("参数错误");

            var sum = 0d;
            for (int i = 0; i < input.Item1.Length; i++)
            {
                sum -= input.Item2[i] * Math.Log(input.Item1[i]);
            }

            return sum;
        }

        public double Differentiate((int, double[], double[]) input)
        {
            return -input.Item3[input.Item1] / input.Item2[input.Item1];
        }
    }
}
