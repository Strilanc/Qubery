using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MoreLinq;

[TestClass]
public class BiMatTest {
    [TestMethod]
    public void TestMultiplication() {
        var m1 = new BiMat(new Complex[,] {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
        var m2 = new BiMat(new Complex[,] {{17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}});
        var actual = m1*m2;
        var expected = new BiMat(new Complex[,] {{250, 260, 270, 280}, {618, 644, 670, 696}, {986, 1028, 1070, 1112}, {1354, 1412, 1470, 1528}});
        Assert.AreEqual(expected, actual);
    }
    [TestMethod]
    public void TestToString() {
        var i = Complex.ImaginaryOne;
        var m = new BiMat(new[,] {
            { 0, 1, -1, i },
            { -i, 1+i, 1-i/2, 0.5 + 0.25*i },
            { 0.75 + 0.75*i, 0, 2, 3 + 4*i },
            { 5, 7, 11, 1000009 }
        });
        var e = "" + Environment.NewLine +
                "┌                       ┐" + Environment.NewLine +
                "│    0   1   -1       i │" + Environment.NewLine +
                "│   -i 1+i 1-½i    ½+¼i │" + Environment.NewLine +
                "│ ¾+¾i   0    2    3+4i │" + Environment.NewLine +
                "│    5   7   11 1000009 │" + Environment.NewLine +
                "└                       ┘" + Environment.NewLine;
        Assert.AreEqual(e, m.ToString());
    }
    [TestMethod]
    public void TestRowPhaseEquality() {
        var i = Complex.ImaginaryOne;
        var m1 = new BiMat(new Complex[,] {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        });
        var m2 = new BiMat(new Complex[,] {
            {1, 0, 0, 0},
            {0, i, 0, 0},
            {0, 0, -1, 0},
            {0, 0, 0, -i}
        });
        var m3 = new BiMat(new[,] {
            {0, 0, 0, 0},
            {0, i, 0, 0},
            {0, 0, -1, 0},
            {0, 0, 0, -i}
        });
        var m4 = new BiMat(new[,] {
            {0, 0, 0, 0},
            {0, i, 0, 0},
            {0, 0, -1, 0},
            {0, 0, 0, -i}
        });
        var m5 = new BiMat(new[,] {
            {0, 0, 0, 0},
            {1, i, 1, 2},
            {0, 0, -1, 0},
            {0, 0, 0, -i}
        });
        var m6 = new BiMat(new[,] {
            {0, 0, 0, 0},
            {-1, -i, -1, -2},
            {0, 0, -1, 0},
            {0, 0, 0, -i}
        });
        Assert.IsTrue(
            new[] {m1, m2, m3, m4, m5, m6}
            .DistinctBy(e => e, new BiMat.RowPhaseInsensitiveComparer())
            .SequenceEqual(new[] {m1, m3, m5}));
    }
}
