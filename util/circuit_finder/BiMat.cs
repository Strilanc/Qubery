using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public struct BiMat {
    public readonly Complex[,] Cells;

    public BiMat(Complex[,] cells) {
        this.Cells = cells;
    }

    private static readonly Complex i = Complex.ImaginaryOne;
    public static Complex[,] SqrtX = new[,] {
        {(1+i)/2, (1-i)/2},
        {(1-i)/2, (1+i)/2}
    };
    public static Complex[,] SqrtY = new[,] {
        {(1+i)/2, (-1-i)/2},
        {(1+i)/2, (1+i)/2}
    };
    public static Complex[,] SqrtZ = new[,] {
        {1, 0},
        {0, i}
    };

    public static readonly BiMat I = new BiMat(new Complex[,] {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    });
    public static BiMat On2Controlled(Complex[,] cells) {
        var a = cells[0, 0];
        var b = cells[0, 1];
        var c = cells[1, 0];
        var d = cells[1, 1];
        return new BiMat(new[,] {
            {1, 0, 0, 0},
            {0, a, 0, b},
            {0, 0, 1, 0},
            {0, c, 0, d}
        });
    }

    public static BiMat On1Controlled(Complex[,] cells) {
        var a = cells[0, 0];
        var b = cells[0, 1];
        var c = cells[1, 0];
        var d = cells[1, 1];
        return new BiMat(new[,] {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, a, b},
            {0, 0, c, d}
        });
    }

    public static BiMat On2(Complex[,] cells) {
        var a = cells[0, 0];
        var b = cells[0, 1];
        var c = cells[1, 0];
        var d = cells[1, 1];
        return new BiMat(new[,] {
            {a, 0, b, 0},
            {0, a, 0, b},
            {c, 0, d, 0},
            {0, c, 0, d}
        });
    }

    public class RowPhaseInsensitiveComparer : IEqualityComparer<BiMat> {
        public bool Equals(BiMat x, BiMat y) {
            for (var r = 0; r < 4; r++) {
                Complex factorX = 1;
                Complex factorY = 1;
                for (var c = 0; c < 4; c++) {
                    if (x.Cells[r, c] != 0) {
                        factorX = x.Cells[r, c];
                        factorY = y.Cells[r, c];
                        if (factorY == 0) {
                            return false;
                        }
                        break;
                    }
                }
                for (var c = 0; c < 4; c++) {
                    if (x.Cells[r, c]*factorY != y.Cells[r, c]*factorX) {
                        return false;
                    }
                }
            }
            return true;
        }
        public int GetHashCode(BiMat obj) {
            var hash = 0;
            for (var r = 0; r < 4; r++) {
                Complex f = 1;
                for (var c = 0; c < 4; c++) {
                    if (obj.Cells[r, c] != 0) {
                        f = Complex.Conjugate(obj.Cells[r, c]);
                        break;
                    }
                }

                for (var c = 0; c < 4; c++) {
                    unchecked {
                        hash *= 31;
                        hash += (obj.Cells[r, c] * f).GetHashCode();
                    }
                }
            }
            return hash;
        }
    }
    public static BiMat On1(Complex[,] cells) {
        var a = cells[0, 0];
        var b = cells[0, 1];
        var c = cells[1, 0];
        var d = cells[1, 1];
        return new BiMat(new[,] {
            {a, b, 0, 0},
            {c, d, 0, 0},
            {0, 0, a, b},
            {0, 0, c, d}
        });
    }

    public Complex RowDot(BiMat m2, int r1, int r2) {
        var t = Complex.Zero;
        for (var c = 0; c < 4; c++) {
            t += Cells[r1, c]*m2.Cells[r2, c];
        }
        return t;
    }

    public static BiMat operator *(BiMat m1, BiMat m2) {
        var m3 = new Complex[4, 4];
        for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
                for (var k = 0; k < 4; k++) {
                    m3[r, c] += m1.Cells[r, k] * m2.Cells[k, c];
                }
            }
        }
        return new BiMat(m3);
    }

    public BiMat Pow(int power) {
        if (power <= 0) throw new ArgumentOutOfRangeException("power", "power <= 0");
        var t = this;
        while (power > 1) {
            t *= this;
            power -= 1;
        }
        return t;
    }

    public static bool operator ==(BiMat m1, BiMat m2) {
        for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
                if (m1.Cells[r, c] != m2.Cells[r, c]) {
                    return false;
                }
            }
        }
        return true;
    }
    public static bool operator !=(BiMat m1, BiMat m2) {
        return !(m1 == m2);
    }
    public override bool Equals(object obj) {
        return obj is BiMat && ((BiMat)obj) == this;
    }
    public override int GetHashCode() {
        var hash = 0;
        for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
                unchecked {
                    hash *= 31;
                    hash += Cells[r, c].GetHashCode();
                }
            }
        }
        return hash;
    }

    private static String RealToString(double d) {
        if (d == 0.5) return "½";
        if (d == -0.5) return "-½";
        if (d == 0.25) return "¼";
        if (d == -0.25) return "-¼";
        if (d == 0.75) return "¾";
        if (d == -0.75) return "-¾";
        return d + "";
    }
    private static String ComplexToString(Complex c) {
        if (c == (1 + i) / 2) return "↗";
        if (c == (1 - i) / 2) return "↘";
        if (c == (-1 + i) / 2) return "↖";
        if (c == (-1 - i) / 2) return "↙";
        if (c == 0.5) return "→";
        if (c == i / 2) return "↑";
        if (c == -0.5) return "←";
        if (c == -i / 2) return "↓";
        if (c == 0) return "0";
        if (c == 1) return "1";
        if (c == -1) return "-1";
        if (c == Complex.ImaginaryOne) return "i";
        if (c == -Complex.ImaginaryOne) return "-i";

        if (c.Real == 0) return RealToString(c.Imaginary) + "i";
        if (c.Imaginary == 0) return RealToString(c.Real);

        if (c.Imaginary == 1) return RealToString(c.Real) + "+i";
        if (c.Imaginary == -1) return RealToString(c.Real) + "-i";
        if (c.Imaginary > 0) return RealToString(c.Real) + "+" + RealToString(c.Imaginary) + "i";
        return RealToString(c.Real) + "-" + RealToString(-c.Imaginary) + "i";
    }

    public override string ToString() {
        var cells = Cells;
        var rows = (from r in Enumerable.Range(0, 4)
                    select (from c in Enumerable.Range(0, 4)
                            select ComplexToString(cells[r, c])
                            ).ToArray()
                    ).ToArray();
        var maxWidths = Enumerable.Range(0, 4).Select(c => 
            Enumerable.Range(0, 4).Select(r => 
                rows[r][c].Length).Max()).ToArray();
        var padded = (from r in Enumerable.Range(0, 4)
                      select (from c in Enumerable.Range(0, 4)
                              select rows[r][c].PadLeft(maxWidths[c])
                              ).ToArray()
                    ).ToArray();

        var first = Environment.NewLine + "┌ " + new String(' ', maxWidths.Sum() + 3) + " ┐" + Environment.NewLine;
        var last = Environment.NewLine + "└ " + new String(' ', maxWidths.Sum() + 3) + " ┘" + Environment.NewLine;
        return first + string.Join(
            Environment.NewLine,
            Enumerable.Range(0, 4).Select(r =>
                "│ " + string.Join(
                    " ",
                    Enumerable.Range(0, 4).Select(c =>
                        padded[r][c])) + " │")) + last;
    }
}
