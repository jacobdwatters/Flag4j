package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixElemMultTests {

    Matrix A, B, result, expResult;
    CMatrix BC, resultC, expResultC;
    double[][] entriesA, entriesB;
    CNumber[][] entriesBC;

    private double[] getExp(double[] src1, double[] src2) {
        double[] result = new double[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src1[i]*src2[i];
        }

        return result;
    }

    private CNumber[] getExp(double[] src1, CNumber[] src2) {
        CNumber[] result = new CNumber[src1.length];

        for(int i=0; i<result.length; i++) {
            result[i] = src2[i].mult(src1[i]);
        }

        return result;
    }


    @Test
    void elemMultTest() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94, -0.4442}, {0.0000234, 1333.4, 44.5, 134.3}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);
        expResult = new Matrix(A.shape.clone(), getExp(A.entries, B.entries));

        result = A.elemMult(B);

        assertArrayEquals(expResult.entries, result.entries);
        assertEquals(expResult.shape, result.shape);

        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesB = new double[][]{{4.344, 555.6, 94}, {0.0000234, 1333.4, 44.5}};
        A = new Matrix(entriesA);
        B = new Matrix(entriesB);

        assertThrows(IllegalArgumentException.class, ()->A.elemMult(B));
    }


    @Test
    void elemMultComplexTest() {
        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new CNumber[][]{{new CNumber(1.4, 5), new CNumber(0, -1), new CNumber(1.3), new CNumber()},
                {new CNumber(4.55, -93.2), new CNumber(-2, -13), new CNumber(8.9), new CNumber(0, 13)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);
        expResultC = new CMatrix(A.shape.clone(), getExp(A.entries, BC.entries));

        resultC = A.elemMult(BC);

        assertArrayEquals(expResultC.entries, resultC.entries);
        assertEquals(expResultC.shape, resultC.shape);

        // ----------------- Sub-case 1 -----------------
        entriesA = new double[][]{{1, 2, -3.324, 13.44}, {4, 5, -6, 0}};
        entriesBC = new CNumber[][]{{new CNumber(1.4, 5), new CNumber(0, -1), new CNumber(1.3)},
                {new CNumber(4.55, -93.2), new CNumber(-2, -13), new CNumber(8.9)}};
        A = new Matrix(entriesA);
        BC = new CMatrix(entriesBC);

        assertThrows(IllegalArgumentException.class, ()->A.elemMult(BC));
    }
}
