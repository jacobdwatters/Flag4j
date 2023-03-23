package com.flag4j.matrix;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixElementScalarTests {
    double[][] aEntries, expEntries;
    CNumber[][] expEntriesC;
    double scalar;
    CNumber scalarC;
    Matrix A;
    Matrix expResult;
    CMatrix expResultC;


    @Test
    void scalDivTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.334/1.44, -2.3112/1.44, 334.3/1.44},
                {4.13/1.44, -35.33/1.44, 6/1.44}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.entries, A.scalDiv(scalar).entries);
        assertEquals(expResult.shape, A.scalDiv(scalar).shape);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalarC = new CNumber(1.3245, -42.5);
        expEntriesC = new CNumber[][]{{new CNumber(1.334).div(scalarC),
                new CNumber(-2.3112).div(scalarC),
                new CNumber(334.3).div(scalarC)},
                {new CNumber(4.13).div(scalarC),
                        new CNumber(-35.33).div(scalarC),
                        new CNumber(6).div(scalarC)}};
        expResultC = new CMatrix(expEntriesC);

        assertArrayEquals(expResultC.entries, A.scalDiv(scalarC).entries);
        assertEquals(expResultC.shape, A.scalDiv(scalarC).shape);
    }


    @Test
    void recipTest() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.0/1.334, 1.0/-2.3112, 1.0/334.3},
                {1.0/4.13, 1.0/-35.33, 1.0/6}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.entries, A.recip().entries);
        assertEquals(expResult.shape, A.recip().shape);
    }
}
