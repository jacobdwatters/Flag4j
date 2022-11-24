package com.flag4j;


import com.flag4j.complex_numbers.CNumber;
import com.flag4j.concurrency.algorithms.subtraction.ConcurrentSubtraction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixSubTests {

    Matrix A, B, actDiff, expDiff;
    CMatrix BComplex, expComplex, actComplex;
    double[][] aEntries, bEntries, expDiffEntries;
    double b;
    CNumber bComplex;
    CNumber[][] expComplexEntries, bComplexEntries;


    @Test
    void subTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{4, 10, -23}, {7.5, 937, 44}, {1123.4323, -0.421, 9.0002}};
        expDiffEntries = new double[][] {{-3, -8, 26}, {-3.5, -932, -38}, {-1116.4323, 8.421, -1.9999999999953388E-4}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actDiff = A.sub(B);
        assertArrayEquals(expDiff.entries, actDiff.entries);

        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, 24}, {6.34, Double.POSITIVE_INFINITY}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};
        expDiffEntries = new double[][] {{Double.NEGATIVE_INFINITY, Double.NaN},
                {Double.NEGATIVE_INFINITY, Double.NaN}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actDiff = A.sub(B);
        assertArrayEquals(expDiff.entries, actDiff.entries);
    }


    @Test
    void concurrentSubTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{4, 10, -23}, {7.5, 937, 44}, {1123.4323, -0.421, 9.0002}};
        expDiffEntries = new double[][] {{-3, -8, 26}, {-3.5, -932, -38}, {-1116.4323, 8.421, -1.9999999999953388E-4}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actDiff = ConcurrentSubtraction.sub(A, B);
        assertArrayEquals(expDiff.entries, actDiff.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, 24}, {6.34, Double.POSITIVE_INFINITY}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};
        expDiffEntries = new double[][] {{Double.NEGATIVE_INFINITY, Double.NaN},
                {Double.NEGATIVE_INFINITY, Double.NaN}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actDiff = ConcurrentSubtraction.sub(A, B);
        assertArrayEquals(expDiff.entries, actDiff.entries);
    }


    @Test
    void doubleSubTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        b = 451.394;
        expDiffEntries = new double[][] {{-450.394, -449.394, -448.394}, {-447.394, -446.394, -445.394}, {-444.394, -443.394, -442.394}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);

        actDiff = A.sub(b);
        assertArrayEquals(expDiff.entries, actDiff.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34, Double.POSITIVE_INFINITY}};
        b = 23.84;
        expDiffEntries = new double[][] {{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34-23.84, Double.POSITIVE_INFINITY}};

        expDiff = new Matrix(expDiffEntries);
        A = new Matrix(aEntries);

        actDiff = A.sub(b);
        assertArrayEquals(expDiff.entries, actDiff.entries);
    }


    @Test
    void complexSubTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}};
        bComplex = new CNumber(451.394, -0.245);
        expComplexEntries = new CNumber[][]{{new CNumber("-450.394+0.245i"), new CNumber("-449.394+0.245i")},
                {new CNumber("-447.394+0.245i"), new CNumber("-446.394+0.245i")},
                {new CNumber("-444.394+0.245i"), new CNumber("-443.394+0.245i")}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);

        actComplex = A.sub(bComplex);

        assertArrayEquals(expComplex.entries, actComplex.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34, Double.POSITIVE_INFINITY}};
        bComplex = new CNumber(4, 2);
        expComplexEntries = new CNumber[][]{{new CNumber(Double.NEGATIVE_INFINITY, -2), new CNumber(Double.NaN, -2)},
                {new CNumber(6.34-4, -2), new CNumber(Double.POSITIVE_INFINITY, -2)}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);

        actComplex = A.sub(bComplex);

        for(int i=0; i<actComplex.numRows(); i++) {
            for(int j=0; j<actComplex.numCols(); j++) {
                if(i==0 && j==1) {
                    assertTrue(Double.isNaN(actComplex.entries[i][j].re));
                    assertEquals(-2, actComplex.entries[i][j].im);
                } else {
                    assertEquals(expComplex.entries[i][j], actComplex.entries[i][j]);
                }
            }
        }
    }

    @Test
    void subCMatrixTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bComplexEntries = new CNumber[][]{{new CNumber("2-0.123i"), new CNumber("5.662+342.25i"), new CNumber("18+2i")},
                {new CNumber("67i"), new CNumber("25"), new CNumber("-435.3494i")},
                {new CNumber("1852 + 2.3i"), new CNumber("-15.4"), new CNumber("-0.24+2i")}};
        expComplexEntries = new CNumber[][]{{new CNumber("-1+0.123i"), new CNumber("-3.662-342.25i"), new CNumber("-15-2i")},
                {new CNumber("4-67i"), new CNumber("-20"), new CNumber("6+435.3494i")},
                {new CNumber("-1845-2.3i"), new CNumber("23.4"), new CNumber("9.24-2i")}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);

        actComplex = A.sub(BComplex);
        assertArrayEquals(expComplex.entries, actComplex.entries);
    }
}
