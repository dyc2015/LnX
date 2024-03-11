using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
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
            Init();
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            TrainData = null;
            TestData = null;
        }

        ConvolutionalNeuralNetwork convolutionalNeuralNetwork;

        private void LeanBtn_Click(object sender, RoutedEventArgs e)
        {
            var data = TrainData.Take(100);

            new Thread(() =>
            {
                var log = "开始训练...\n";
                Dispatcher.Invoke(() =>
                {
                    LoadingBar.IsIndeterminate = true;
                    TrainLog.Text = log;
                });

                //var count = 0;
                convolutionalNeuralNetwork.Train(data.Select(x => x.Pixels), data.Select(x => x.Labels), () =>
                {
                    var costs = convolutionalNeuralNetwork.Costs;
                    log += $"第{costs.Count}轮，误差：{costs.LastOrDefault()}\n";
                    //count++;

                    //if (count > 100 && count % 100 == 0)
                    //{
                        Dispatcher.Invoke(() =>
                        {
                            TrainLog.Text = log;
                            TrainLog.ScrollToEnd();

                            var sources = ImageSourceUtil.CreateBitmapSource(convolutionalNeuralNetwork.Transformers[0].Output);
                            TestImg.Source = sources[0];
                            TestImg1.Source = sources[1];
                            TestImg2.Source = sources[2];
                            sources = ImageSourceUtil.CreateBitmapSource(convolutionalNeuralNetwork.Transformers[1].Output);
                            TestImg3.Source = sources[0];
                            TestImg4.Source = sources[1];
                            TestImg5.Source = sources[2];
                        });
                    //}
                });

                Dispatcher.Invoke(() =>
                {
                    LoadingBar.IsIndeterminate = false;

                    var len = convolutionalNeuralNetwork.Costs.Count;
                    var values = new List<ObservablePoint>();
                    for (var i = 0; i < len; i+=len / 5)
                    {
                        values.Add(new ObservablePoint(i, convolutionalNeuralNetwork.Costs[i]));
                    }

                    MyChart.Series =
                    [
                        new LineSeries<ObservablePoint>
                        {
                            Values = values
                        }
                    ];
                });
            }).Start();
        }

        private void TestBtn_Click(object sender, RoutedEventArgs e)
        {
            convolutionalNeuralNetwork.Context.Labels = null;
            convolutionalNeuralNetwork.Compute(TrainData[0].Pixels);

            var t = convolutionalNeuralNetwork.Output;
        }

        public void Init()
        {
            convolutionalNeuralNetwork = ConvolutionalNeuralNetworkBuilder.Create()
                .SetBatchSize(10)
                .SetAlpha(0.1)
                .SetMaxEpcoh(100)
                .Append(new ConvolutionalTransformer(KernelUtil.Create(5, 5, 3), Function.CreateReLU()))
                .Append(new PoolingTransformer(3, 3, Function.CreateMaxPooling()))
                .Append(new FullyConnectTransformer(DeepNeuralNetworkBuilder.Create().SetLayerConfig(192, 10, 10).Build()))
                .Build();
        }
    }
}
