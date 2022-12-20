package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixElementScalarTests {
    double[][] aEntries, expEntries;
    CNumber[][] expEntriesC;
    double scalar;
    CNumber scalarC;
    Matrix A;
    Matrix expResult;
    CMatrix expResultC;


    @Test
    void scalDivTests() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.334/1.44, -2.3112/1.44, 232.15277777777777},
                {2.8680555555555554, -35.33/1.44, 4.166666666666666}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.entries, A.scalDiv(scalar).entries);
        assertEquals(expResult.shape, A.scalDiv(scalar).shape);

        // -------------- Sub-case 2 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalarC = new CNumber(1.3245, -44.5);
        expEntriesC = new CNumber[][]{{new CNumber("0.000891462747975438 + 0.029950994552591162i"),
                new CNumber("-0.0015444892826992744 - 0.05189110840325988i"),
                new CNumber("0.22340029733747294 + 7.505710254071383i")},
                {new CNumber(4.13).div(scalarC),
                        new CNumber(-35.33).div(scalarC),
                        new CNumber(6).div(scalarC)}};
        expResultC = new CMatrix(expEntriesC);

        assertArrayEquals(expResultC.entries, A.scalDiv(scalarC).entries);
        assertEquals(expResultC.shape, A.scalDiv(scalarC).shape);
    }


    @Test
    void recepTests() {
        // -------------- Sub-case 1 --------------
        aEntries = new double[][]{{1.334, -2.3112, 334.3}, {4.13, -35.33, 6}};
        A = new Matrix(aEntries);
        scalar = 1.44;
        expEntries = new double[][]{{1.0/1.334, 1.0/-2.3112, 1.0/334.3},
                {1.0/4.13, 1.0/-35.33, 1.0/6}};
        expResult = new Matrix(expEntries);

        assertArrayEquals(expResult.entries, A.recep().entries);
        assertEquals(expResult.shape, A.recep().shape);
    }
}
