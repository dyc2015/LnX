using LnX.ML.CNN;
using LnX.ML.DNN;
using LnX.ML.Utils;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace LnX.ML
{
    /// <summary>
    /// LeanWindow.xaml 的交互逻辑
    /// </summary>
    public partial class LeanWindow : Window
    {
        /// <summary>
        /// 手写字数据集合
        /// </summary>
        public MinstData[] TrainData, TestData;

        public LeanWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Test();
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            TrainData = null;
            TestData = null;
        }

        public void Test()
        {
            var context = new TransformContext();
            var transformer = new ConvolutionalTransformer(KernelUtil.Create(5, 5), Function.CreateReLU());
            var poolingTransformer = new PoolingTransformer(transformer, 3, 3, Function.CreateMaxPooling());
            var fullyConnectTransformer = new FullyConnectTransformer(poolingTransformer,
                DeepNeuralNetworkBuilder.Create()
                .SetLayerConfig(64, 10, 10)
                .SetActivationFunction(Function.CreateReLU())
                .SetErrorFunction(Function.CreateCrossEntropy())
                .Build());
            for (int i = 0; i < 10; i++)
            {
                context.Input = TrainData[i].Pixels;
                context.Labels = TrainData[i].Labels;

                transformer.Transform(context);

                var sources = ImageSourceUtil.CreateBitmapSource(transformer.Output);
                TestImg.Source = sources[0];
                //TestImg1.Source = sources[1];
                //TestImg2.Source = sources[2];

                sources = ImageSourceUtil.CreateBitmapSource(poolingTransformer.Output);
                TestImg3.Source = sources[0];
                //TestImg4.Source = sources[1];
                //TestImg5.Source = sources[2];

                fullyConnectTransformer.BackPropagation(context);

                var t = fullyConnectTransformer.Output;
            }
        }
    }
}
