﻿using LiveChartsCore.Defaults;
using LnX.ML.CNN;
using LnX.ML.Utils;
using System;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML.DNN
{
    public class DeepNeuralNetworkBuilder
    {
        public static DeepNeuralNetworkBuilder Create() => new();

        ///// <summary>
        ///// 层级配置
        ///// </summary>
        //public int[] LayerConfig => layerConfig;
        int[] layerConfig;
        /// <summary>
        /// 设置分层配置
        /// </summary>
        /// <param name="layerConfig"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public DeepNeuralNetworkBuilder SetLayerConfig(params int[] layerConfig)
        {
            if (layerConfig.Length < 2) throw new Exception("层数不能低于2");

            this.layerConfig = layerConfig;
            return this;
        }

        ILossFunction errorFunction;
        /// <summary>
        /// 设置误差函数
        /// </summary>
        /// <param name="function"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetErrorFunction(ILossFunction function)
        {
            errorFunction = function;
            return this;
        }

        IActivationFunction activationFunction;
        /// <summary>
        /// 设置激活函数
        /// </summary>
        /// <param name="function"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetActivationFunction(IActivationFunction function)
        {
            activationFunction = function;
            return this;
        }

        int batchSize = 1;
        /// <summary>
        /// 设置更新权重批大小
        /// </summary>
        /// <param name="batchSize"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetBatchSize(int batchSize)
        {
            this.batchSize = batchSize;
            return this;
        }

        double alpha = 0.001;
        /// <summary>
        /// 设置学习率
        /// </summary>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetAlpha(double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        int maxEpcoh = 10000;
        /// <summary>
        /// 设置训练最大循环次数，大于这个次数将结束训练
        /// </summary>
        /// <param name="maxEpcoh"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetMaxEpcoh(int maxEpcoh)
        {
            this.maxEpcoh = maxEpcoh;
            return this;
        }

        double minError = 0.001;
        /// <summary>
        /// 设置最小误差，小于这个误差将结束训练
        /// </summary>
        /// <param name="minError"></param>
        /// <returns></returns>
        public DeepNeuralNetworkBuilder SetMinError(double minError)
        {
            this.minError = minError;
            return this;
        }

        public DeepNeuralNetwork Build()
        {
            if (layerConfig == null || layerConfig.Length == 0)
                throw new Exception("分层配置有误");

            var neurons = new INeuron[layerConfig.Length][];
            var inputNeurons = new InputNeuron[layerConfig[0] + 1];
            var outputNeurons = new OutputNeuron[layerConfig[^1]];
            var random = new Random();
            var function = activationFunction ?? Function.CreateReLU();
            //var normalFunction = Function.CreateNormal();

            for (int i = 0; i < layerConfig.Length; i++)
            {
                var isStart = i == 0;
                var isEnd = i == layerConfig.Length - 1;

                var currentLen = layerConfig[i];
                var lastIndex = i - 1;
                var lasetLen = isStart ? 0 : layerConfig[lastIndex];
                var nextLen = isEnd ? 0 : layerConfig[i + 1];

                neurons[i] = isStart ? inputNeurons : isEnd ? outputNeurons : (new INeuron[currentLen + 1]);

                for (int j = 0; j < currentLen; j++)
                {
                    if (isStart)
                    {
                        inputNeurons[j] = new InputNeuron(new Synapse[nextLen]);
                        continue;
                    }

                    var frontSynapses = new Synapse[lasetLen + 1];

                    INeuron currentNeuron;
                    if (isEnd)
                    {
                        outputNeurons[j] = new OutputNeuron(function, frontSynapses);
                        currentNeuron = outputNeurons[j];
                    }
                    else
                    {
                        currentNeuron = new HiddenNeuron(function, frontSynapses, new Synapse[nextLen]);
                    }

                    for (int k = 0; k <= lasetLen; k++)
                    {
                        var lastNeuron = neurons[lastIndex][k];
                        var synapse = new Synapse(random.NextDouble(), lastNeuron, currentNeuron);
                        lastNeuron.RearSynapses[j] = synapse;
                        frontSynapses[k] = synapse;
                    }

                    neurons[i][j] = currentNeuron;
                }

                if (!isEnd)
                {
                    var biasNeuron = new InputNeuron(new Synapse[nextLen], true);
                    neurons[i][currentLen] = biasNeuron;

                    if (isStart)
                    {
                        inputNeurons[currentLen] = biasNeuron;
                    }
                }
            }

            return new DeepNeuralNetwork(inputNeurons, outputNeurons, errorFunction ?? Function.CreateCrossEntropy(),
                neurons, batchSize, alpha, maxEpcoh, minError);
        }
    }

    /// <summary>
    /// 深度神经网络
    /// </summary>
    /// <param name="inputNeurons"></param>
    /// <param name="outputNeuron"></param>
    /// <param name="errorFunction"></param>
    public class DeepNeuralNetwork(InputNeuron[] inputNeurons,
        OutputNeuron[] outputNeuron,
        ILossFunction errorFunction,
        INeuron[][] neurons,
        int batchSize, double alpha,
        int maxEpcoh, double minError)
    {

        /// <summary>
        /// 每批数量
        /// </summary>
        readonly int batchSize = batchSize;

        /// <summary>
        /// 最小误差
        /// </summary>
        readonly double minError = minError;

        /// <summary>
        /// 最大循环次数
        /// </summary>
        readonly int maxEpcoh = maxEpcoh;

        /// <summary>
        /// softmax
        /// </summary>
        readonly SoftmaxFunction softmaxFunction = new();

        /// <summary>
        /// 误差函数对每个输出求导
        /// </summary>
        readonly double[] deriveFromOutput = new double[outputNeuron.Length];

        /// <summary>
        /// 学习率
        /// </summary>
        double alpha = alpha;

        /// <summary>
        /// 神经元结构信息
        /// </summary>
        public INeuron[][] Neurons => neurons;

        /// <summary>
        /// 输入神经元
        /// </summary>
        public InputNeuron[] InputNeurons { get; } = inputNeurons;

        /// <summary>
        /// 输出神经元
        /// </summary>
        public OutputNeuron[] OutputNeurons { get; } = outputNeuron;

        /// <summary>
        /// 误差函数
        /// </summary>
        public ILossFunction ErrorFunction { get; } = errorFunction;

        /// <summary>
        /// 输出值
        /// </summary>
        public double[] Output { get; } = new double[outputNeuron.Length];

        /// <summary>
        /// 输出值（经softmax计算后）
        /// </summary>
        public double[] SoftmaxOutput { get; } = new double[outputNeuron.Length];

        /// <summary>
        /// 每次计算的误差列表
        /// </summary>
        public List<double> Errors = [];

        double cost;
        /// <summary>
        /// 当次训练代价
        /// </summary>
        public double Cost => cost;

        /// <summary>
        /// 单次训练
        /// </summary>
        /// <param name="input"></param>
        /// <param name="labels"></param>
        public void Train(double[] input, double[] labels)
        {
            Compute(input);
            ComputeCost(labels);
        }

        /// <summary>
        /// 根据传入数据训练
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="labels"></param>
        public void Train(double[][] datas, double[][] labels, Action? onEpcohEnd = null)
        {
            Errors.Clear();

            double tmpCost = 0, lastCost = -1;
            int epcoh = 0, dataLen = datas.GetLength(0);
            while (epcoh < maxEpcoh)
            {
                for (int i = 0; i < dataLen; i++)
                {
                    var len = i + 1;
                    var isUpdate = len == dataLen || (len >= batchSize && len % batchSize == 0);

                    Train(datas[i], labels[i]);

                    tmpCost += cost;

                    if (isUpdate) BackPropagation();
                }

                tmpCost /= dataLen;

                Errors.Add(tmpCost);

                onEpcohEnd?.Invoke();

                if (Math.Abs(tmpCost) <= minError) return;

                if (tmpCost == lastCost) return;//梯度消失

                if (lastCost != -1)
                {
                    if (tmpCost - lastCost > 0)
                    {//误差变大减小学习率
                        alpha /= 2;
                    }
                }

                lastCost = tmpCost;
                tmpCost = 0;

                epcoh++;
            }
        }

        /// <summary>
        /// 计算输出
        /// </summary>
        /// <param name="input"></param>
        public void Compute(double[] input)
        {
            SetInput(input);

            for (int i = 0; i < Output.Length; i++)
            {
                OutputNeurons[i].ComputeOutput();
                Output[i] = OutputNeurons[i].Output;
            }

            for (int i = 0; i < Output.Length; i++)
            {
                SoftmaxOutput[i] = softmaxFunction.Compute((i, Output));
            }
        }

        public void ComputeCost(double[] labels)
        {
            for (var i = 0; i < Output.Length; i++)
            {
                deriveFromOutput[i] = ErrorFunction.Differentiate((i, SoftmaxOutput, labels));//dL/ds
            }

            for (var i = 0; i < Output.Length; i++)
            {
                OutputNeurons[i].RecordedError(softmaxFunction.Differentiate((i, SoftmaxOutput, deriveFromOutput)));//ds/da(l)
            }

            cost = ErrorFunction.Compute((SoftmaxOutput, labels));
        }

        /// <summary>
        /// 进行反向传播
        /// </summary>
        public void BackPropagation()
        {
            neurons.ForEach(x => x.ResetError());//重置误差

            for (var i = 0; i < Output.Length; i++)
            {
                OutputNeurons[i].UpdateError(OutputNeurons[i].AvgError);//ds/da(l)
                OutputNeurons[i].ClearErrorRecords();
            }

            foreach (var outputNeuron in OutputNeurons)
            {
                outputNeuron.UpdateWeight(alpha);
            }
        }

        void SetInput(double[] input)
        {
            if (input == null || input.Length != InputNeurons.Length - 1)
                throw new ArgumentException("参数长度有误", nameof(input));

            for (int i = 0; i < input.Length; i++)
            {
                InputNeurons[i].Input = input[i];
            }
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < neurons.GetLength(0); i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    sb.Append(neurons[i][j].ToString());
                    sb.Append('\t');
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }
    }

    public interface INeuron
    {
        /// <summary>
        /// 输入
        /// </summary>
        double Input { get; }
        /// <summary>
        /// 输出
        /// </summary>
        double Output { get; }
        /// <summary>
        /// 误差
        /// </summary>
        double Error { get; }

        /// <summary>
        /// 前一层的突触
        /// </summary>
        Synapse[] FrontSynapses { get; }

        /// <summary>
        /// 后一层的突触
        /// </summary>
        Synapse[] RearSynapses { get; }

        /// <summary>
        /// 更新误差
        /// </summary>
        /// <param name="error"></param>
        void UpdateError(double error);

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="alpha"></param>
        void UpdateWeight(double alpha);

        /// <summary>
        /// 计算输出
        /// </summary>
        void ComputeOutput();

        /// <summary>
        /// 重置误差
        /// </summary>
        void ResetError();
    }

    public class Synapse(double weight, INeuron frontNeuron, INeuron rearNeuron)
    {
        /// <summary>
        /// 权重
        /// </summary>
        public double Weight => weight;
        private double weight = weight;

        /// <summary>
        /// 前一层神经元
        /// </summary>
        public INeuron FrontNeuron { get; } = frontNeuron;

        /// <summary>
        /// 后一层神经元
        /// </summary>
        public INeuron RearNeuron { get; } = rearNeuron;

        /// <summary>
        /// 更新权重
        /// </summary>
        public void UpdateWeight(double alpha)
        {
            weight -= alpha * RearNeuron.Error * FrontNeuron.Output;//alpha * dL/da(l) * da(l)/dz(l) * dz(l)/dw(l)
        }
    }

    /// <summary>
    /// 隐藏神经元
    /// </summary>
    /// <param name="activationFunction"></param>
    /// <param name="frontSynapses"></param>
    /// <param name="rearSynapses"></param>
    public class HiddenNeuron(IActivationFunction activationFunction,
        Synapse[] frontSynapses,
        Synapse[] rearSynapses) : INeuron
    {
        double input = 0;
        public double Input => input;

        double output = 0;
        public double Output => output;

        double error = 0;
        public double Error => error;

        public IActivationFunction ActivationFunction { get; } = activationFunction;

        public Synapse[] FrontSynapses { get; } = frontSynapses;
        public Synapse[] RearSynapses { get; } = rearSynapses;

        public virtual void UpdateError(double error)
        {
            error *= ActivationFunction.Differentiate(input);//dL/da(l) * da(l)/dz(l)
            this.error += error;

            foreach (var x in FrontSynapses)
            {
                x.FrontNeuron.UpdateError(error * x.Weight);//dL/da(l) * da(l)/dz(l) * dz(l)/da(l-1)
            }
        }

        public virtual void UpdateWeight(double alpha)
        {
            foreach (var x in FrontSynapses)
            {
                x.UpdateWeight(alpha);
            }
        }

        public void ComputeOutput()
        {
            input = 0;
            foreach (Synapse synapse in FrontSynapses)
            {
                synapse.FrontNeuron.ComputeOutput();
                input += synapse.FrontNeuron.Output * synapse.Weight;
            }

            output = ActivationFunction.Compute(input);
        }

        public void ResetError()
        {
            error = 0;

            foreach (var synapse in FrontSynapses)
            {
                synapse.FrontNeuron.ResetError();
            }
        }

        public override string ToString()
        {
            return $"{nameof(HiddenNeuron)}:{Input}|{Output}({string.Join(",", RearSynapses.Select(x => x.Weight))})";
        }
    }

    /// <summary>
    /// 输入神经元
    /// </summary>
    /// <param name="rearSynapses"></param>
    public class InputNeuron(Synapse[] rearSynapses, bool isBias = false, double input = 1) : INeuron
    {
        public double Input { get; set; } = input;
        public double Output => Input;

        private double error = 0;
        public double Error => error;
        public Synapse[] FrontSynapses { get; } = [];
        public Synapse[] RearSynapses { get; } = rearSynapses;

        /// <summary>
        /// 是否Bias
        /// </summary>
        public bool IsBias { get; } = isBias;

        public void UpdateError(double error)
        {
            this.error += error;
        }

        public void UpdateWeight(double alpha)
        {

        }

        public void ComputeOutput()
        {

        }

        public void ResetError()
        {
            error = 0;
        }

        public override string ToString()
        {
            return $"{nameof(InputNeuron)}:{Input}({string.Join(",", RearSynapses.Select(x => x.Weight))})";
        }
    }

    /// <summary>
    /// 输出神经元
    /// </summary>
    /// <param name="activationFunction"></param>
    /// <param name="frontSynapses"></param>
    public class OutputNeuron(IActivationFunction activationFunction,
        Synapse[] frontSynapses) : HiddenNeuron(activationFunction, frontSynapses, [])
    {
        double errorSum = 0;
        int errorCount = 0;

        public double AvgError => errorSum / errorCount;

        /// <summary>
        /// 记录误差
        /// </summary>
        /// <param name="error"></param>
        public void RecordedError(double error)
        {
            errorSum += error;
            errorCount++;
        }

        /// <summary>
        /// 清除误差记录
        /// </summary>
        public void ClearErrorRecords()
        {
            errorSum = 0;
            errorCount = 0;
        }

        public override string ToString()
        {
            return $"{nameof(OutputNeuron)}:{Input}|{Output}";
        }
    }
}
