package com.flag4j.operations.dense_sparse.real_complex;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexDenseSparseMatMultTests {

    static double[][] rdEntries = {};
    static CNumber[][] cdEntries = {};

    static double[] rspEntries = {};
    static CNumber[] cspEntries ={};

    static int[][] rindices;
    static int[][] cindices;

    static Shape rspShape;
    static Shape cspShape;

    static Matrix realDense;
    static CMatrix complexDense;
    static SparseMatrix realSp;
    static SparseCMatrix complexSp;

    static CNumber[][] expEntries;
    static CMatrix exp;

    @BeforeAll
    static void setup() {
        rdEntries = new double[][]{
                {1.234, -00.024, 0, 1},
                {100.4, 5.14, -1.444, 0.041},
                {1.45, 985.1, -75.1, 4}
        };
        cdEntries = new CNumber[][]{
                {new CNumber(-0.24, 14.5), new CNumber(0.425)},
                {new CNumber(8.33, -84.4), new CNumber()},
                {new CNumber(4.5, -9.24), new CNumber(0, -85.2)},
                {new CNumber("1.345"), new CNumber("-85.445+15.5i")}
        };

        rspShape = new Shape(2, 3);
        cspShape = new Shape(4, 2);

        rindices = new int[][]{{0, 0, 1}, {0, 2, 1}};
        cindices = new int[][]{{0, 2, 3}, {0, 1, 1}};

        rspEntries = new double[]{1.445, -9.25, 1.5};
        cspEntries = new CNumber[]{new CNumber(51.5, -0.42),
            new CNumber(-5.25, 15), new CNumber(3.45)};

        realDense = new Matrix(rdEntries);
        complexDense = new CMatrix(cdEntries);
        realSp = new SparseMatrix(rspShape, rspEntries, rindices[0], rindices[1]);
        complexSp = new SparseCMatrix(cspShape, cspEntries, cindices[0], cindices[1]);
    }


    @Test
    void realDenseComplexSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        expEntries = new CNumber[][]{{new CNumber("63.551-0.51828i"), new CNumber("3.45")},
                {new CNumber("5170.6-42.168i"), new CNumber("7.722449999999999-21.66i")},
                {new CNumber("74.675-0.609i"), new CNumber("408.075-1126.5i")}};
        exp = new CMatrix(expEntries);
        assertEquals(exp, realDense.mult(complexSp));
    }


    @Test
    void complexDenseRealSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        expEntries = new CNumber[][]{{new CNumber("-0.3468+20.9525i"), new CNumber("0.6375"), new CNumber("2.2199999999999998-134.125i")},
                {new CNumber("12.036850000000001-121.95800000000001i"), new CNumber("0.0"), new CNumber("-77.0525+780.7i")},
                {new CNumber("6.5025-13.3518i"), new CNumber("0.0-127.80000000000001i"), new CNumber("-41.625+85.47i")},
                {new CNumber("1.943525"), new CNumber("-128.1675+23.25i"), new CNumber("-12.44125")}};
        exp = new CMatrix(expEntries);
        assertEquals(exp, complexDense.mult(realSp));
    }
}
