using LnX.ML.CNN;
using LnX.ML.DNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LnX.ML
{
    public static class TransformerUtil
    {
        /// <summary>
        /// 设置更新权重批大小
        /// </summary>
        /// <param name="batchSize"></param>
        /// <returns></returns>
        public static TransformContext SetBatchSize(this TransformContext context, int batchSize)
        {
            context.BatchSize = batchSize;
            return context;
        }

        /// <summary>
        /// 设置学习率
        /// </summary>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static TransformContext SetAlpha(this TransformContext context, double alpha)
        {
            context.Alpha = alpha;
            return context;
        }

        /// <summary>
        /// 设置训练最大循环次数，大于这个次数将结束训练
        /// </summary>
        /// <param name="maxEpcoh"></param>
        /// <returns></returns>
        public static TransformContext SetMaxEpcoh(this TransformContext context, int maxEpcoh)
        {
            context.MaxEpcoh = maxEpcoh;
            return context;
        }

        /// <summary>
        /// 设置最小误差，小于这个误差将结束训练
        /// </summary>
        /// <param name="minError"></param>
        /// <returns></returns>
        public static TransformContext SetMinError(this TransformContext context, double minError)
        {
            context.MinError = minError;
            return context;
        }
    }
}
