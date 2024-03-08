using Microsoft.VisualStudio.TestTools.UnitTesting;
using LnX.ML.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LnX.ML.CNN;

namespace Lnx.Tests
{
    [TestClass()]
    public class CollectUtilTests
    {
        [TestMethod()]
        public void FillTest()
        {
            var array = new double[2 * 2 * 3 * 2];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            var n = (i * 2 * 2 * 2) + (j * 2 * 2) + (k * 2) + l;
                            array[n] = n;
                        }
                    }
                }
            }

            var t = CollectUtil.FillNew(array, 2, 2, 3, 2);

            Assert.IsNotNull(t);
        }
    }
}