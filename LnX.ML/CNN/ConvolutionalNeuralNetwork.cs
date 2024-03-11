using LnX.ML.DNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;

namespace LnX.ML.CNN
{

    public class ConvolutionalNeuralNetworkBuilder
    {
        public static ConvolutionalNeuralNetworkBuilder Create() => new();

        public ConvolutionalNeuralNetwork Build()
        {
            if (transformers.Count == 0) 
                throw new Exception("转换器配置为空");
            if (!typeof(FullyConnectTransformer).IsAssignableFrom(transformers.Last().GetType())) 
                throw new Exception("末级转换器需为全连接转换器");

            var context = new TransformContext();
            context.SetMaxEpcoh(maxEpcoh);
            context.SetAlpha(alpha);
            context.SetBatchSize(batchSize);
            context.SetMinError(minError);

            return new ConvolutionalNeuralNetwork(transformers, context);
        }

        readonly List<ITransformer> transformers = [];
        public ConvolutionalNeuralNetworkBuilder Append(ITransformer transformer)
        {
            var lastTransformer = transformers.LastOrDefault();
            if(lastTransformer != null)
            {
                transformer.FrontTransformer = lastTransformer;
                lastTransformer.RearTransformer = transformer;
            }

            transformers.Add(transformer);

            return this;
        }

        double alpha = 0.001;
        public ConvolutionalNeuralNetworkBuilder SetAlpha(double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        int batchSize = 1;
        public ConvolutionalNeuralNetworkBuilder SetBatchSize(int batchSize)
        {
            this.batchSize = batchSize;
            return this;
        }

        int maxEpcoh = 1000;
        public ConvolutionalNeuralNetworkBuilder SetMaxEpcoh(int maxEpcoh)
        {
            this.maxEpcoh = maxEpcoh;
            return this;
        }

        double minError = 0.001;
        public ConvolutionalNeuralNetworkBuilder SetMinError(double minError)
        {
            this.minError = minError;
            return this;
        }
    }
    /// <summary>
    /// 卷积神经网络
    /// </summary>
    public class ConvolutionalNeuralNetwork(List<ITransformer> transformers, TransformContext context)
    {
        public List<ITransformer> Transformers { get; } = transformers;
        public ITransformer InputTransformer { get; } = transformers.First();
        public FullyConnectTransformer OutputTransformer { get; } = (FullyConnectTransformer)transformers.Last();
        public TransformContext Context { get; } = context;

        public List<double> Costs = [];

        public ITensor Output => OutputTransformer.Output;

        public void Train(ITensor tensor, double[] labels)
        {
            Context.Labels = labels;
            Compute(tensor);

            Context.Input = null;
            Context.Labels = null;
        }

        public void Compute(ITensor tensor)
        {
            Context.Input = tensor;
            InputTransformer.Transform(Context);
        }

        public void Train(IEnumerable<ITensor> tensors, 
            IEnumerable<double[]> labels,
            Action onEpcohEnd = null)
        {
            Costs.Clear();

            double tmpCost = 0, lastCost = -1;
            int epcoh = 0, dataLen = tensors.Count(),
                batchSize = Context.BatchSize;
            while (epcoh < Context.MaxEpcoh)
            {
                for (int i = 0; i < dataLen; i++)
                {
                    Train(tensors.ElementAt(i), labels.ElementAt(i));

                    tmpCost += OutputTransformer.Cost;

                    var len = i + 1;
                    if (len == dataLen || (len >= batchSize && len % batchSize == 0)) 
                        OutputTransformer.BackPropagation(Context);
                }

                tmpCost /= dataLen;

                Costs.Add(tmpCost);

                onEpcohEnd?.Invoke();

                if (Math.Abs(tmpCost) <= Context.MinError) return;

                if (tmpCost == lastCost) return;//梯度消失

                if (lastCost != -1 && tmpCost - lastCost > 0)//误差变大减小学习率
                {
                    Context.Alpha /= 2;
                }

                lastCost = tmpCost;
                tmpCost = 0;

                epcoh++;
            }
        }
    }
}
