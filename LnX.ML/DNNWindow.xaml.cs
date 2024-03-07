using LiveChartsCore.SkiaSharpView;
using LiveChartsCore;
using System;
using System.Collections.Generic;
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
using System.ComponentModel;
using LiveChartsCore.Defaults;
using LnX.ML.DNN;

namespace LnX.ML
{
    /// <summary>
    /// DNNWindow.xaml 的交互逻辑
    /// </summary>
    public partial class DNNWindow : Window
    {
        public DNNWindow()
        {
            InitializeComponent();
        }

        double[][] datas;
        double[][] labels;
        private void GenerateBtn_Click(object sender, RoutedEventArgs e)
        {
            var btn = (Button)sender;
            btn.IsEnabled = false;

            var viewData = (ViewData)DataContext;
            var count = viewData.DataCount;

            datas = new double[count][];
            labels = new double[count][];

            var random = new Random();
            for (int i = 0; i < count; i++)
            {
                var x = random.NextDouble() * 5 + 2;
                var y = random.NextDouble() * 5 + 2;

                datas[i] = [x, y];

                if (y >= x)
                {
                    labels[i] = [0, 1];
                }
                else
                {
                    labels[i] = [1, 0];
                }
            }

            var observablePoints1 = new List<ObservablePoint>();
            var observablePoints2 = new List<ObservablePoint>();

            for (int i = 0; i < datas.GetLength(0); i++)
            {
                if (labels[i][0] == 0)
                {
                    observablePoints1.Add(new ObservablePoint
                    {
                        X = datas[i][0],
                        Y = datas[i][1]
                    });
                }
                else
                {
                    observablePoints2.Add(new ObservablePoint
                    {
                        X = datas[i][0],
                        Y = datas[i][1]
                    });
                }

                btn.IsEnabled = true;
            }

            viewData.Series =
            [
                new ScatterSeries<ObservablePoint>
                {
                    Values = observablePoints1,
                    Name = "分类1",
                },
                new ScatterSeries<ObservablePoint>
                {
                    Values = observablePoints2,
                    Name = "分类2",
                }
            ];
        }

        DeepNeuralNetwork deepNeuralNetwork;
        private void TrainBtn_Click(object sender, RoutedEventArgs e)
        {
            deepNeuralNetwork = DeepNeuralNetworkBuilder.Create()
            .SetLayerConfig(2, 2)
            .SetMaxEpcoh(1000)
            .Build();

            new Thread(() =>
            {
                var log = "开始训练...\n";
                Dispatcher.Invoke(() =>
                {
                    TrainBtn.IsEnabled = false;
                    LoadingBar.IsIndeterminate = true;
                    TrainLog.Text = log;
                });

                var count = 0;
                deepNeuralNetwork.Train(datas, labels, () =>
                {
                    var errors = deepNeuralNetwork.Errors;
                    log += $"第{errors.Count}轮，误差：{errors.LastOrDefault()}\n";
                    count++;

                    if (count % 100 == 0)
                    {
                        Dispatcher.Invoke(() =>
                        {
                            TrainLog.Text = log;
                            TrainLog.ScrollToEnd();
                        });
                    }
                });

                Dispatcher.Invoke(() =>
                {
                    TrainLog.Text = log;
                    TrainLog.ScrollToEnd();

                    TrainBtn.IsEnabled = true;
                    LoadingBar.IsIndeterminate = false;
                });
            }).Start();
        }

        private void TestBtn_Click(object sender, RoutedEventArgs e)
        {
            var viewData = (ViewData)DataContext;

            var neurons = deepNeuralNetwork.Neurons;
            double w11 = neurons[0][0].RearSynapses[0].Weight,
                w12 = neurons[0][0].RearSynapses[1].Weight,
                w21 = neurons[0][1].RearSynapses[0].Weight,
                w22 = neurons[0][1].RearSynapses[1].Weight,
                b1 = neurons[0][2].RearSynapses[0].Weight,
                b2 = neurons[0][2].RearSynapses[1].Weight;

            var series = viewData.Series;
            viewData.Series = [];

            series.AddRange(new List<LineSeries<ObservablePoint>>
            {
                new()
                {
                    Values = [new (0, (b2 - b1) / (w21 - w22)), new (7, (7 * (w12 - w11) + b2 - b1) / (w21 - w22))],
                    Fill = null,
                    LineSmoothness = 0
                }
            });

            viewData.Series = series;

            deepNeuralNetwork.Compute([2, 3]);

            var str = deepNeuralNetwork.ToString();
            var output = deepNeuralNetwork.SoftmaxOutput;

            TrainLog.Text += $"网络结构：{str}， 预测结果：{string.Join(",", output)}\n";
        }
    }

    public class ViewData : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        private List<ISeries> series = [];
        public List<ISeries> Series
        {
            get => series;
            set
            {
                series = value;
                OnPropertyChanged(nameof(Series));
            }
        }

        public int DataCount { get; set; } = 100;

        private void OnPropertyChanged(string name)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }

    }
}
