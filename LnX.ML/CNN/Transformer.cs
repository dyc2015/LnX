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
        /// 后一个转换器传过来的误差
        /// </summary>
        public ITensor? Error { get; set; }
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

    public abstract class TransformerBase(ITransformer frontTransformer) : ITransformer
    {
        public ITensor Output => output;
        protected ITensor output;

        public ITensor Input => input;
        protected ITensor input;

        public ITensor Error => error;
        protected ITensor error;

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

        public void Transform(TransformContext context)
        {
            input = FrontTransformer?.Output ?? context.Input;

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
            if (output != null && output.Width == width && output.Height == height
                && output.Num == num && output.Dimension == dimension) return;

            output = new Tensor(width, height, num, dimension);
        }

        public abstract void DoTransform(TransformContext context);
        public abstract void BackPropagation(TransformContext context);
    }

    /// <summary>
    /// 卷积转换器
    /// </summary>
    /// <param name="kernel">卷积核</param>
    /// <param name="stride"></param>
    public class ConvolutionalTransformer(ITensor kernel,
        IDifferentiableFunction<double> function,
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
        readonly IDifferentiableFunction<double> function = function;

        public override void DoTransform(TransformContext context)
        {
            if (input.Dimension != Kernel.Dimension)
                throw new ArgumentException("卷积核维度与输入数据维度不符");

            int width = (input.Width - Kernel.Width + 1) / Stride,
                height = (input.Height - Kernel.Height + 1) / Stride;

            SetOutput(width, height, Kernel.Num);

            for (int rn = 0; rn < Kernel.Num; rn++)
                for (int rw = 0; rw < width; rw++)
                    for (int rh = 0; rh < height; rh++)
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

                                output[rn, 0, rw, rh] = function.Compute(sum);
                            }
        }

        public override void BackPropagation(TransformContext context)
        {
            if (input == null) throw new Exception("inputRecord为空");

            var error = RearTransformer.Error;
            for (int i = 0; i < error.Num; i++)
            {

            }
        }
    }

    public class PoolingTransformer(ITransformer front,
        int width, int height,
        IFunction<double[,]> function) : TransformerBase(front)
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

        public override void DoTransform(TransformContext context)
        {
            if (input.Width % Width != 0 || input.Height % Height != 0)
                throw new Exception("数据宽高有误");

            int width = input.Width / Width, height = input.Height / Height;

            SetOutput(width, height, input.Num, input.Dimension);

            var subData = new double[Width, Height];

            for (int n = 0; n < input.Num; n++)
                for (int d = 0; d < input.Dimension; d++)
                    for (int w = 0; w < width; w++)
                        for (var h = 0; h < height; h++)
                        {
                            for (int w1 = 0; w1 < Width; w1++)
                            {
                                for (int h1 = 0; h1 < Height; h1++)
                                {
                                    subData[w1, h1] = input[n, d, w * Width + w1, h * Height + h1];
                                }
                            }

                            output[n, d, w, h] = Function.Compute(subData);
                        }
        }

        public override void BackPropagation(TransformContext context)
        {

        }

    }

    /// <summary>
    /// 全连接转换
    /// </summary>
    /// <param name="outputNum">输出结果数量</param>
    public class FullyConnectTransformer(ITransformer front,
        DeepNeuralNetwork deepNeuralNetwork) : TransformerBase(front)
    {
        public override void BackPropagation(TransformContext context)
        {
            deepNeuralNetwork.BackPropagation();
            var errors = deepNeuralNetwork.InputNeurons.Select(x => x.Error);

            error ??= new Tensor(input.Width, input.Height, input.Num, input.Dimension);

            errors.FillTo(error);

            FrontTransformer.BackPropagation(context);
        }

        public override void DoTransform(TransformContext context)
        {
            SetOutput(context.Labels.Length, 1);

            var flattenInput = CollectUtil.Flatten(input);

            deepNeuralNetwork.Train(flattenInput, context.Labels);

            for (int i = 0; i < deepNeuralNetwork.Output.Length; i++)
            {
                output[0, 0, 0, i] = deepNeuralNetwork.Output[i];
            }
        }
    }
}
