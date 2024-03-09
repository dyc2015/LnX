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
    /// 转换器上下文
    /// </summary>
    /// <param name="input"></param>
    /// <param name="labels"></param>
    public class TransformContext()
    {
        /// <summary>
        /// 原始输入
        /// </summary>
        public ITensor Input { get; set; }
        /// <summary>
        /// 实际值
        /// </summary>
        public double[] Labels { get; set; }

        /// <summary>
        /// 学习率
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// 每批数量
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// 最小误差
        /// </summary>
        public double MinError { get; set; }

        /// <summary>
        /// 最大循环次数
        /// </summary>
        public int MaxEpcoh { get; set; }
    }

    /// <summary>
    /// 转换器
    /// </summary>
    public interface ITransformer
    {
        /// <summary>
        /// 输入
        /// </summary>
        ITensor Input { get; }

        /// <summary>
        /// 输出
        /// </summary>
        ITensor Output { get; }

        /// <summary>
        /// 误差信息
        /// </summary>
        ITensor Error { get; }

        /// <summary>
        /// 前一个转换器
        /// </summary>
        ITransformer FrontTransformer { get; set; }

        /// <summary>
        /// 后一个转换器
        /// </summary>
        ITransformer RearTransformer { get; set; }

        /// <summary>
        /// 转换数据
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        void Transform(TransformContext context);

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="context"></param>
        void BackPropagation(TransformContext context);
    }

    public abstract class TransformerBase : ITransformer
    {
        public ITensor Output => output;
        protected ITensor output;

        public ITensor Input => input;
        protected ITensor input;

        public ITensor Error => error;
        protected ITensor error;

        ITransformer frontTransformer;
        public ITransformer FrontTransformer
        {
            get => frontTransformer;
            set
            {
                frontTransformer = value;

                if (value != null)
                {
                    value.RearTransformer = this;
                }
            }
        }
        public ITransformer RearTransformer { get; set; }

        public TransformerBase(ITransformer frontTransformer)
        {
            this.frontTransformer = frontTransformer;
            if (frontTransformer != null)
            {
                frontTransformer.RearTransformer = this;
            }
        }

        public void Transform(TransformContext context)
        {
            input = FrontTransformer?.Output ?? context.Input;
            error ??= new Tensor(input.Width, input.Height, input.Num, input.Dimension);

            DoTransform(context);

            RearTransformer?.Transform(context);
        }

        /// <summary>
        /// 设置输出
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="num"></param>
        /// <param name="dimension"></param>
        protected void SetOutput(int width, int height, int num = 1, int dimension = 1)
        {
            //if (output != null && output.Width == width && output.Height == height
            //    && output.Num == num && output.Dimension == dimension) return;

            output ??= new Tensor(width, height, num, dimension);
        }

        public void BackPropagation(TransformContext context)
        {
            DoBackPropagation(context);
            frontTransformer?.BackPropagation(context);
        }

        protected abstract void DoTransform(TransformContext context);
        protected abstract void DoBackPropagation(TransformContext context);
    }

    /// <summary>
    /// 卷积转换器
    /// </summary>
    /// <param name="kernel">卷积核</param>
    /// <param name="stride"></param>
    public class ConvolutionalTransformer(ITensor kernel,
        IActivationFunction function,
        int stride = 1,
        ITransformer front = null) : TransformerBase(front)
    {
        /// <summary>
        /// 卷积核
        /// </summary>
        public ITensor Kernel { get; } = kernel;

        /// <summary>
        /// 步幅
        /// </summary>
        public int Stride { get; } = stride;

        /// <summary>
        /// 激活函数
        /// </summary>
        readonly IActivationFunction function = function;

        ITensor derivativeValues;

        protected override void DoTransform(TransformContext context)
        {
            if (input.Dimension != Kernel.Dimension)
                throw new ArgumentException("卷积核维度与输入数据维度不符");

            int width = (input.Width - Kernel.Width + 1) / Stride,
                height = (input.Height - Kernel.Height + 1) / Stride;

            SetOutput(width, height, Kernel.Num);
            derivativeValues ??= new Tensor(output.Width, output.Height, output.Num, output.Dimension);

            for (int rn = 0; rn < Kernel.Num; rn++)
                for (int rw = 0; rw < width; rw++)
                    for (int rh = 0; rh < height; rh++)
                        for (int kw = 0; kw < Kernel.Width; kw++)
                            for (int kh = 0; kh < Kernel.Height; kh++)
                            {
                                var sum = 0d;
                                for (int kd = 0; kd < Kernel.Dimension; kd++)
                                {
                                    for (int ni = 0; ni < input.Num; ni++)
                                    {
                                        sum += Kernel[rn, kd, kw, kh] * input[ni, kd, kw + rw * Stride, kh + rh * Stride];
                                    }
                                }

                                output[rn, 0, rw, rh] = function.Compute(sum);
                                derivativeValues[rn, 0, rw, rh] = function.Differentiate(sum);
                            }
        }

        protected override void DoBackPropagation(TransformContext context)
        {
            int width = (input.Width - Kernel.Width + 1) / Stride,
                height = (input.Height - Kernel.Height + 1) / Stride;

            for (int rn = 0; rn < Kernel.Num; rn++)
                for (int rw = 0; rw < width; rw++)
                    for (int rh = 0; rh < height; rh++)
                        for (int kw = 0; kw < Kernel.Width; kw++)
                            for (int kh = 0; kh < Kernel.Height; kh++)
                            {
                                var curError = derivativeValues[rn, 0, rw, rh] * RearTransformer.Error[rn, 0, rw, rh];
                                for (int kd = 0; kd < Kernel.Dimension; kd++)
                                {
                                    for (int ni = 0; ni < input.Num; ni++)
                                    {
                                        int w = kw + rw * Stride, h = kh + rh * Stride;
                                        Kernel[rn, kd, kw, kh] -= context.Alpha * curError * input[ni, kd, w, h];
                                        error[ni, kd, w, h] = curError;
                                    }
                                }
                            }
        }
    }

    public class PoolingTransformer(ITransformer front,
        int width, int height,
        IPoolingFunction function) : TransformerBase(front)
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
        public IPoolingFunction Function { get; } = function;

        double[,,,][,] mapping;

        protected override void DoTransform(TransformContext context)
        {
            if (input.Width % Width != 0 || input.Height % Height != 0)
                throw new Exception("数据宽高有误");

            int width = input.Width / Width, height = input.Height / Height;

            SetOutput(width, height, input.Num, input.Dimension);

            mapping ??= new double[input.Num, input.Dimension, width, height][,];

            for (int n = 0; n < input.Num; n++)
                for (int d = 0; d < input.Dimension; d++)
                    for (int w = 0; w < width; w++)
                        for (var h = 0; h < height; h++)
                        {
                            mapping[n, d, w, h] ??= new double[Width, Height];

                            for (int w1 = 0; w1 < Width; w1++)
                            {
                                for (int h1 = 0; h1 < Height; h1++)
                                {
                                    mapping[n, d, w, h][w1, h1] = input[n, d, w * Width + w1, h * Height + h1];
                                }
                            }

                            output[n, d, w, h] = Function.Compute(mapping[n, d, w, h]);
                        }
        }

        protected override void DoBackPropagation(TransformContext context)
        {
            for (int n = 0; n < output.Num; n++)
                for (int d = 0; d < output.Dimension; d++)
                    for (int w = 0; w < output.Width; w++)
                        for (var h = 0; h < output.Height; h++)
                        {
                            var tmpError = Function.Differentiate(mapping[n, d, w, h]);
                            for (int w1 = 0; w1 < Width; w1++)
                            {
                                for (int h1 = 0; h1 < Height; h1++)
                                {
                                    error[n, d, w * Width + w1, h * Height + h1] = tmpError[w1, h1] * RearTransformer.Error[n, d, w, h];
                                }
                            }
                        }
        }
    }

    /// <summary>
    /// 全连接转换
    /// </summary>
    /// <param name="outputNum">输出结果数量</param>
    public class FullyConnectTransformer(ITransformer front,
        DeepNeuralNetwork deepNeuralNetwork) : TransformerBase(front)
    {
        protected override void DoBackPropagation(TransformContext context)
        {
            deepNeuralNetwork.BackPropagation();

            deepNeuralNetwork.InputNeurons.Where(x => !x.IsBias).Select(x => x.Error).FillTo(Error);
        }

        protected override void DoTransform(TransformContext context)
        {
            SetOutput(context.Labels.Length, 1);

            var flattenInput = CollectUtil.Flatten(input);

            deepNeuralNetwork.Train(flattenInput, context.Labels);

            for (int i = 0; i < deepNeuralNetwork.Output.Length; i++)
            {
                output[0, 0, 0, i] = deepNeuralNetwork.SoftmaxOutput[i];
            }
        }
    }
}
