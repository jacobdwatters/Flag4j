package org.flag4j.matrix;

import org.flag4j.CustomAssertions;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MatrixElemOppTests {

    static double[][] aEntries;
    static Matrix A;

    @BeforeAll
    static void setup() {
        aEntries = new double[][]{
                {1.123, 2, -0.01, 0.0, -0.0, 15},
                {104.51, -64, 54, 100.455, 0.00024, 1024},
                {-9.2245, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, 10.4, -158.14}
        };
        A = new Matrix(aEntries);
    }


    @Test
    void sqrtComplexTestCase() {
        CNumber[][] expEntries;
        CMatrix exp;

        // --------------------- Sub-case 1 ---------------------
        expEntries = new CNumber[][]{
                {CNumber.sqrt(1.123), CNumber.sqrt(2), CNumber.sqrt(-0.01), CNumber.sqrt(0.0), CNumber.sqrt(-0.0), CNumber.sqrt(15)},
                {CNumber.sqrt(104.51), CNumber.sqrt(-64), CNumber.sqrt(54), CNumber.sqrt(100.455), CNumber.sqrt(0.00024), CNumber.sqrt(1024)},
                {CNumber.sqrt(-9.2245), CNumber.sqrt(Double.POSITIVE_INFINITY), CNumber.sqrt(Double.NEGATIVE_INFINITY), CNumber.sqrt(Double.NaN), CNumber.sqrt(10.4), CNumber.sqrt(-158.14)}
        };
        exp = new CMatrix(expEntries);
        CustomAssertions.assertEqualsNaN(exp, A.sqrtComplex());
    }
}
