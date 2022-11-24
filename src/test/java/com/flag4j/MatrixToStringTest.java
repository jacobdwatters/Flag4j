package com.flag4j;

import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixToStringTest {
    double[][] aEntries;
    Matrix A;
    String exp, act;

    @Test
    void toStringTest() {
        PrintOptions.resetAll();

        // ------------- Sub-case 1 -----------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new Matrix(aEntries);
        exp =   "[ [ 1  2  3 ]\n" +
                "  [ 4  5  6 ]\n" +
                "  [ 7  8  9 ] ]";
        act = A.toString();

        assertEquals(exp, act);

        // ------------- Sub-case 2 -----------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, Double.NaN}, {-923.42, 9239.4}, {954.9, 1}};
        A = new Matrix(aEntries);
        exp =   "[ [ -Infinity   NaN   ]\n" +
                "  [  -923.42   9239.4 ]\n" +
                "  [   954.9      1    ] ]";
        act = A.toString();

        assertEquals(exp, act);

        // ------------- Sub-case 3 -----------------
        A = new Matrix(30, 30, 6);
        exp =   "[ [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ]\n" +
                "   ...\n" +
                "  [ 6  6  6  6  6  6  6  6  6  6  ... 6  ] ]";
        act = A.toString();

        assertEquals(exp, act);

        // ------------- Sub-case 3 -----------------
        A = new Matrix();
        exp =   "[[]]";
        act = A.toString();

        assertEquals(exp, act);
    }
}
