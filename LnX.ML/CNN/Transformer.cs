using LnX.ML.DNN;
using LnX.ML.Utils;
using Newtonsoft.Json;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML.CNN
{
    /// <summary>
    /// 转换器
    /// </summary>
    public interface ITransformer
    {
        /// <summary>
        /// 转换数据
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        void Transform(TransformContext context);
    }

    public class TransformContext(ITensor input, double[] labels)
    {
        /// <summary>
        /// 原始输入
        /// </summary>
        public ITensor Input { get; set; } = input;
        /// <summary>
        /// 实际值
        /// </summary>
        public double[] Labels { get; set; } = labels;

        /// <summary>
        /// 上一个转换器转换之后的值
        /// </summary>
        public ITensor Output { get => output ?? Input; set => output = value; }
        private ITensor? output;
    }

    /// <summary>
    /// 卷积转换器
    /// </summary>
    /// <param name="kernel">卷积核</param>
    /// <param name="stride"></param>
    public class ConvolutionalTransformer(ITensor kernel, IDifferentiableFunction<double> function, int stride = 1) : ITransformer
    {
        /// <summary>
        /// 卷积核
        /// </summary>
        public ITensor Kernel { get; } = kernel;

        /// <summary>
        /// 激活函数
        /// </summary>
        public IDifferentiableFunction<double> Function { get; } = function;

        /// <summary>
        /// 步幅
        /// </summary>
        public int Stride { get; } = stride;

        public void Transform(TransformContext context)
        {
            var input = context.Output;
            if (input.Dimension != Kernel.Dimension) throw new ArgumentException("卷积核维度与输入数据维度不符");

            int width = (input.Width - Kernel.Width) / Stride, height = (input.Height - Kernel.Height) / Stride;
            var result = new Tensor(width, height, Kernel.Num);

            for (int rn = 0; rn < Kernel.Num; rn++)
                for (int rh = 0; rh < height; rh++)
                    for (int rw = 0; rw < width; rw++)
                        for (int kw = 0; kw < Kernel.Width; kw++)
                            for (int kh = 0; kh < Kernel.Height; kh++)
                            {
                                var sum = 0d;
                                for (int ni = 0; ni < input.Num; ni++)
                                {
                                    for (int kd = 0; kd < Kernel.Dimension; kd++)
                                    {
                                        sum += Kernel[rn, kd, kw, kh] * input[ni, kd, kw + rw * Stride, kh + rh * Stride];
                                    }
                                }

                                result[rn, 0, rw, rh] = Function.Compute(sum);
                            }

            context.Output = result;
        }
    }

    public class PoolingTransformer(int width, int height, IFunction<double[,]> function) : ITransformer
    {
        /// <summary>
        /// 池化窗口宽度
        /// </summary>
        public int Width { get; } = width;
        /// <summary>
        /// 池化窗口高度
        /// </summary>
        public int Height { get; } = height;
        /// <summary>
        /// 池化函数
        /// </summary>
        public IFunction<double[,]> Function { get; } = function;

        public void Transform(TransformContext context)
        {
            var input = context.Output;
            int width = input.Width / Width, height = input.Height / Height;
            var result = new Tensor(width, height, input.Num, input.Dimension);

            for (int i = 0; i < input.Num; i++)
                for (int j = 0; j < input.Dimension; j++)
                    for (int k = 0; k < width; k++)
                        for (var i1 = 0; i1 < height; i1++)
                        {
                            var subData = new double[Width, Height];
                            for (int j1 = 0; j1 < Width; j1++)
                            {
                                for (int k1 = 0; k1 < Height; k1++)
                                {
                                    subData[j1, k1] = input[i, j, k * Width + j1, i1 * Height + k1];
                                }
                            }

                            result[i, j, k, i1] = Function.Compute(subData);
                        }

            context.Output = result;
        }
    }

    /// <summary>
    /// 全连接转换
    /// </summary>
    /// <param name="outputNum">输出结果数量</param>
    public class FullyConnectTransformer() : ITransformer
    {
        public void Transform(TransformContext context)
        {
            var result = new Tensor(context.Labels.Length, 1);
            var flatInput = CollectUtil.Flatten(context.Output);

            var deepNeuralNetwork = DeepNeuralNetworkBuilder.Create()
                .SetLayerConfig(flatInput.Length, 10, 10, context.Labels.Length)
                .SetAlpha(0.001)
                .Build();

            deepNeuralNetwork.Train(flatInput, context.Labels);

            for (int i = 0; i < deepNeuralNetwork.Output.Length; i++)
            {
                result[0][0][i, 0] = deepNeuralNetwork.Output[i];
            }

            context.Output = result;
        }
    }
}
