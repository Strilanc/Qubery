using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MoreLinq;

public static class MainClass {
    /// <summary>
    /// Searches for quantum circuits that win a coordination game.
    /// </summary>
    public static void Main() {
        var ops = (from dir in new[] { 
                       new { op = BiMat.SqrtX, nameA="↓", nameB="X", nameC="↑" }, 
                       new { op = BiMat.SqrtZ, nameA="↺", nameB="Z", nameC="↻" },
                       new { op = BiMat.SqrtY, nameA="→", nameB="Y", nameC="←" }
                   }
                   from sir in new[] {
                       new {dir.op, p = 1, name = dir.nameA, cost = 1}, 
                       new {dir.op, p = 2, name = dir.nameB, cost = 2}, 
                       new {dir.op, p = 3, name = dir.nameC, cost = 1}
                   }
                   from wir in new[] {
                       new {
                           op = BiMat.On1(dir.op).Pow(sir.p), 
                           name1="─"+sir.name+"─", 
                           name2="───", 
                           sir.cost
                       }, 
                       new {
                           op = BiMat.On2(dir.op).Pow(sir.p), 
                           name1="───", 
                           name2="─"+sir.name+"─", 
                           sir.cost
                       }, 
                       new {
                           op = BiMat.On1Controlled(dir.op).Pow(sir.p), 
                           name1="─"+sir.name+"─", 
                           name2="─┴─", 
                           cost=sir.cost+2
                       }, 
                       new {
                           op = BiMat.On2Controlled(dir.op).Pow(sir.p), 
                           name1="─┬─", 
                           name2="─"+sir.name+"─", 
                           cost=sir.cost+2
                       }
                   }
                   select wir).ToArray();
        var I = new { op = BiMat.I, name1 = "───", name2 = "───", cost = 0 };
        ops = new[] { I }.Concat(ops).ToArray();

        var cirs = (from opn in ops.NestDistinct(3, e => e.Select(f => f.op).Aggregate((e1, e2) => e2 * e1))
                    select new {
                        name = String.Join("", opn.Select(e => e.name1)) + Environment.NewLine
                             + String.Join("", opn.Select(e => e.name2)) + Environment.NewLine,
                        op = opn.Select(e => e.op).Aggregate((e1, e2) => e2 * e1),
                        cost = opn.Select(e => e.cost).Sum()
                    })
                    .DistinctBy(e => e.op, new BiMat.RowPhaseInsensitiveComparer())
                    .ToArray();
        var solutions = from row_op_1 in cirs
                        from col_op_1 in cirs
                        where ValidateGameCircuits(1, 1, row_op_1.op, col_op_1.op)
                        from row_op_2 in cirs
                        where ValidateGameCircuits(2, 1, row_op_2.op, col_op_1.op)
                        from col_op_2 in cirs
                        where ValidateGameCircuits(1, 2, row_op_1.op, col_op_2.op)
                        where ValidateGameCircuits(2, 2, row_op_2.op, col_op_2.op)
                        from row_op_3 in cirs
                        where ValidateGameCircuits(3, 1, row_op_3.op, col_op_1.op)
                        where ValidateGameCircuits(3, 2, row_op_3.op, col_op_2.op)
                        from col_op_3 in cirs
                        where ValidateGameCircuits(1, 3, row_op_1.op, col_op_3.op)
                        where ValidateGameCircuits(2, 3, row_op_2.op, col_op_3.op)
                        where ValidateGameCircuits(3, 3, row_op_3.op, col_op_3.op)
                        select new {
                            ops = new[] { row_op_1, row_op_2, row_op_3, col_op_1, col_op_2, col_op_3 },
                            cost = row_op_1.cost + row_op_2.cost + row_op_3.cost + col_op_1.cost + col_op_2.cost + col_op_3.cost
                        };
        var bestSolution = solutions.Take(0).FirstOrDefault();
        foreach (var solution in solutions) {
            var s = string.Join(Environment.NewLine, solution.ops.Select(e => e.name + e.op));
            if (bestSolution == null || solution.cost < bestSolution.cost) {
                bestSolution = solution;
                Debug.WriteLine(s);
                Debug.WriteLine("---------------------");
            }
            Console.WriteLine(s);
        }
    }
    
    /// <param name="row">The row forced by the referee or a dice roll.</param>
    /// <param name="col">The column forced by the referee or a dice roll.</param>
    /// <param name="row_move">The row (1=top, 2=mid, 3=bot) of the cell, within the forced column, to *not* play on. 0 means don't play at all.</param>
    /// <param name="col_move">The column (1=left, 2=mid, 3=right) of the cell, within the forced row, to *not* play on. 0 means don't play at all.</param>
    private static bool GameOutcome(int row, int col, int row_move, int col_move) {
        var row_miss = row_move == col || row_move == 0;
        var col_miss = col_move == row || col_move == 0;
        return row_miss != col_miss;
    }

    private static bool ValidateGameCircuits(int row, int col, BiMat row_op, BiMat col_op) {
        for (var row_move = 0; row_move < 4; row_move++) {
            for (var col_move = 0; col_move < 4; col_move++) {
                var winningMove = GameOutcome(row, col, row_move, col_move);
                if (!winningMove && row_op.RowDot(col_op, row_move, col_move) != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    public static IEnumerable<T[]> Nest<T>(this IEnumerable<T> items, int repeat) {
        if (repeat == 0) {
            return new[] { new T[0] };
        }
        return from e in items.Nest(repeat - 1)
               from x in items
               select e.Concat(x).ToArray();
    }

    public static IEnumerable<TVal[]> NestDistinct<TVal, TKey>(this IEnumerable<TVal> items, int repeat, Func<TVal[], TKey> distinctBy) {
        if (repeat == 0) {
            return new[] { new TVal[0] };
        }
        return (from e in items.NestDistinct(repeat - 1, distinctBy)
                from x in items
                select e.Concat(x).ToArray()
                ).DistinctBy(distinctBy);
    }
}
