using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

[TestClass]
public class MainTest {
    [TestMethod]
    public void TestNest() {
        var expeted = new[] {
            new[] {1, 1, 1},
            new[] {1, 1, 2},
            new[] {1, 1, 3},
            new[] {1, 2, 1},
            new[] {1, 2, 2},
            new[] {1, 2, 3},
            new[] {1, 3, 1},
            new[] {1, 3, 2},
            new[] {1, 3, 3},
            new[] {2, 1, 1},
            new[] {2, 1, 2},
            new[] {2, 1, 3},
            new[] {2, 2, 1},
            new[] {2, 2, 2},
            new[] {2, 2, 3},
            new[] {2, 3, 1},
            new[] {2, 3, 2},
            new[] {2, 3, 3},
            new[] {3, 1, 1},
            new[] {3, 1, 2},
            new[] {3, 1, 3},
            new[] {3, 2, 1},
            new[] {3, 2, 2},
            new[] {3, 2, 3},
            new[] {3, 3, 1},
            new[] {3, 3, 2},
            new[] {3, 3, 3}
        };
        var actual = new[] {1, 2, 3}.Nest(3);
        Assert.IsTrue(expeted.Length == actual.Count());
        Assert.IsTrue(expeted.Zip(actual, Enumerable.SequenceEqual).All(e => e));
    }
}
