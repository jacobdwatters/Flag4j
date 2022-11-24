package com.flag4j;

import com.flag4j.concurrency.algorithms.transpose.ConcurrentTranspose;
import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTransposeTests {

    double[][] aEntries, expATEntries;
    Matrix A, expAT, actAT;

    @Test
    void transposeTest() {
        // ------------ Sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        expATEntries = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = A.T();

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);

        // ------------ Sub-case 2 ------------
        aEntries = new double[][]{{1.234, -2.234}, {4234, 524}, {7.3494, 80.1}, {764, 13}};
        expATEntries = new double[][]{{1.234, 4234, 7.3494, 764}, {-2.234, 524, 80.1, 13}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = A.T();

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);

        // ------------ Sub-case 3 ------------
        aEntries = new double[][]{{16.5}};
        expATEntries = new double[][]{{16.5}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = A.T();

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);
    }


    @Test
    void transposeConcurrentTest() {
        // ------------ Sub-case 1 ------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        expATEntries = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = ConcurrentTranspose.T(A);

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);

        // ------------ Sub-case 2 ------------
        aEntries = new double[][]{{1.234, -2.234}, {4234, 524}, {7.3494, 80.1}, {764, 13}};
        expATEntries = new double[][]{{1.234, 4234, 7.3494, 764}, {-2.234, 524, 80.1, 13}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = ConcurrentTranspose.T(A);

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);

        // ------------ Sub-case 3 ------------
        aEntries = new double[][]{{16.5}};
        expATEntries = new double[][]{{16.5}};

        A = new Matrix(aEntries);
        expAT = new Matrix(expATEntries);
        actAT = ConcurrentTranspose.T(A);

        assertEquals(expAT.getShape(), actAT.getShape());
        assertArrayEquals(expAT.entries, actAT.entries);
    }
}
