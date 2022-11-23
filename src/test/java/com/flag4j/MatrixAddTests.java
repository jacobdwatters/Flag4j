package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.concurrency.algorithms.addition.ConcurrentAddition;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixAddTests {

    Matrix A, B, actSum, expSum;
    CMatrix BComplex, expComplex, actComplex;
    double[][] aEntries, bEntries, expSumEntries;
    double b;
    CNumber bComplex;
    CNumber[][] expComplexEntries, bComplexEntries;


    @Test
    void addTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{4, 10, -23}, {7.5, 937, 44}, {1123.4323, -0.421, 9.0002}};
        expSumEntries = new double[][] {{5, 12, -20}, {11.5, 942, 50}, {1130.4323, 7.579, 18.0002}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actSum = A.add(B);
        assertArrayEquals(expSum.entries, actSum.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, 24}, {6.34, Double.POSITIVE_INFINITY}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};
        expSumEntries = new double[][] {{Double.NaN, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actSum = A.add(B);
        assertArrayEquals(expSum.entries, actSum.entries);
    }


    @Test
    void concurrentAddTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{4, 10, -23}, {7.5, 937, 44}, {1123.4323, -0.421, 9.0002}};
        expSumEntries = new double[][] {{5, 12, -20}, {11.5, 942, 50}, {1130.4323, 7.579, 18.0002}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actSum = ConcurrentAddition.add(A, B);
        assertArrayEquals(expSum.entries, actSum.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, 24}, {6.34, Double.POSITIVE_INFINITY}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};
        expSumEntries = new double[][] {{Double.NaN, Double.NaN}, {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);

        actSum = ConcurrentAddition.add(A, B);
        assertArrayEquals(expSum.entries, actSum.entries);
    }


    @Test
    void doubleAddTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        b = 451.394;
        expSumEntries = new double[][] {{452.394, 453.394, 454.394}, {455.394, 456.394, 457.394}, {458.394, 459.394, 460.394}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);

        actSum = A.add(b);
        assertArrayEquals(expSum.entries, actSum.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34, Double.POSITIVE_INFINITY}};
        b = 23.84;
        expSumEntries = new double[][] {{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34+23.84, Double.POSITIVE_INFINITY}};

        expSum = new Matrix(expSumEntries);
        A = new Matrix(aEntries);

        actSum = A.add(b);
        assertArrayEquals(expSum.entries, actSum.entries);
    }


    @Test
    void complexAddTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2}, {4, 5}, {7, 8}};
        bComplex = new CNumber(451.394, -0.245);
        expComplexEntries = new CNumber[][]{{new CNumber("452.394-0.245i"), new CNumber("453.394-0.245i")},
                {new CNumber("455.394-0.245i"), new CNumber("456.394-0.245i")},
                {new CNumber("458.394-0.245i"), new CNumber("459.394-0.245i")}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);

        actComplex = A.add(bComplex);

        assertArrayEquals(expComplex.entries, actComplex.entries);

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{{Double.NEGATIVE_INFINITY, Double.NaN}, {6.34, Double.POSITIVE_INFINITY}};
        bComplex = new CNumber(4, 2);
        expComplexEntries = new CNumber[][]{{new CNumber(Double.NEGATIVE_INFINITY, 2), new CNumber(Double.NaN, 2)},
                {new CNumber(6.34+4, 2), new CNumber(Double.POSITIVE_INFINITY, 2)}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);

        actComplex = A.add(bComplex);

        for(int i=0; i<actComplex.numRows(); i++) {
            for(int j=0; j<actComplex.numCols(); j++) {
                if(i==0 && j==1) {
                    assertTrue(Double.isNaN(actComplex.entries[i][j].re));
                    assertEquals(2, actComplex.entries[i][j].im);
                } else {
                    assertEquals(expComplex.entries[i][j], actComplex.entries[i][j]);
                }
            }
        }
    }

    @Test
    void addCMatrixTest() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bComplexEntries = new CNumber[][]{{new CNumber("2-0.123i"), new CNumber("5.662+342.25i"), new CNumber("18+2i")},
                {new CNumber("67i"), new CNumber("25"), new CNumber("-435.3494i")},
                {new CNumber("1852 + 2.3i"), new CNumber("-15.4"), new CNumber("-0.24+2i")}};
        expComplexEntries = new CNumber[][]{{new CNumber("3-0.123i"), new CNumber("7.662+342.25i"), new CNumber("21+2i")},
                {new CNumber("4+67i"), new CNumber("30"), new CNumber("6-435.3494i")},
                {new CNumber("1859+2.3i"), new CNumber("-7.4"), new CNumber("8.76+2i")}};

        expComplex = new CMatrix(expComplexEntries);
        A = new Matrix(aEntries);
        BComplex = new CMatrix(bComplexEntries);

        actComplex = A.add(BComplex);
        assertArrayEquals(expComplex.entries, actComplex.entries);
    }
}
