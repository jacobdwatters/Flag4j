package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixSubTests {
    double[][] aEntries, bEntries;
    CNumber[][] bCEntries;

    MatrixOld A, B;
    CMatrixOld BC, expC;
    double b;
    CNumber bC;
    MatrixOld sum, exp;
    CMatrixOld sumC;
    Shape expShape;
    double[] expEntries;
    CNumber[] expEntriesC;

    @Test
    void matrixMatrixTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1.343, 6.7, -88.4}};
        A = new MatrixOld(aEntries);
        B = new MatrixOld(bEntries);
        expShape = A.shape;
        expEntries = new double[]{1-0.333, 2-56.4, 3-13.4, 4+1.44, 0, 6-85.1, 7-1.343, 8-6.7, 9+88.4};
        exp = new MatrixOld(expShape, expEntries);

        sum = A.sub(B);

        assertEquals(exp, sum);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new MatrixOld(aEntries);
        B = new MatrixOld(bEntries);
        expShape = A.shape;
        expEntries = new double[]{1-0.333, 2-56.4, 3-13.4, 4 + 1.44, 0, 6-85.1};
        exp = new MatrixOld(expShape, expEntries);

        sum = A.sub(B);

        assertEquals(exp, sum);

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1, 2, 3}};
        A = new MatrixOld(aEntries);
        B = new MatrixOld(bEntries);
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));

        // --------------- Sub-case 4 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new MatrixOld(aEntries);
        B = new MatrixOld(bEntries);
        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
    }


    @Test
    void matrixDoubleTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        b = 2.133;
        A = new MatrixOld(aEntries);
        expShape = A.shape;
        expEntries = new double[]{1-2.133, 2-2.133, 3-2.133, 4-2.133, 5-2.133, 6-2.133, 7-2.133, 8-2.133, 9-2.133};
        exp = new MatrixOld(expShape, expEntries);

        sum = A.sub(b);

        assertEquals(exp, sum);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        b = 2.133;
        A = new MatrixOld(aEntries);
        expShape = A.shape;
        expEntries = new double[]{1-2.133, 2-2.133, 3-2.133, 4-2.133, 5-2.133, 6-2.133};
        exp = new MatrixOld(expShape, expEntries);

        sum = A.sub(b);

        assertEquals(exp, sum);
    }


    @Test
    void matrixCNumberTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bC = new CNumber(33.444, -9.3545);
        A = new MatrixOld(aEntries);
        expShape = A.shape;
        expEntriesC = new CNumber[]{
                new CNumber(1).sub(bC), new CNumber(2).sub(bC), new CNumber(3).sub(bC),
                new CNumber(4).sub(bC), new CNumber(5).sub(bC), new CNumber(6).sub(bC),
                new CNumber(7).sub(bC), new CNumber(8).sub(bC), new CNumber(9).sub(bC)
        };

        sumC = A.sub(bC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bC = new CNumber(33.444, -9.3545);
        A = new MatrixOld(aEntries);
        expShape = A.shape;
        expEntriesC = new CNumber[]{
                new CNumber(1).sub(bC), new CNumber(2).sub(bC), new CNumber(3).sub(bC),
                new CNumber(4).sub(bC), new CNumber(5).sub(bC), new CNumber(6).sub(bC)
        };

        sumC = A.sub(bC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);
    }


    @Test
    void matrixCMatrixTestCase() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)},
                {new CNumber(1.23), new CNumber(8, 3), new CNumber(9, -0.000000001)}
        };
        A = new MatrixOld(aEntries);
        BC = new CMatrixOld(bCEntries);
        expShape = A.shape;
        expEntriesC = new CNumber[]{
                new CNumber(1-1.23, 344.5), new CNumber(2-2.33, -5.6), new CNumber(3-3.13, 34),
                new CNumber(4, -66.45), new CNumber(5-33.1334, -5513.5), new CNumber(6-99.3),
                new CNumber(7-1.23), new CNumber(0, -3), new CNumber(0, 0.000000001)
        };

        sumC = A.sub(BC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);


        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)}
        };
        A = new MatrixOld(aEntries);
        BC = new CMatrixOld(bCEntries);
        expShape = A.shape;
        expEntriesC = new CNumber[]{
                new CNumber(1-1.23, 344.5), new CNumber(2-2.33, -5.6), new CNumber(3-3.13, 34),
                new CNumber(4, -66.45), new CNumber(5-33.1334, -5513.5), new CNumber(6-99.3)
        };

        sumC = A.sub(BC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)}
        };
        A = new MatrixOld(aEntries);
        BC = new CMatrixOld(bCEntries);

        assertThrows(LinearAlgebraException.class, ()->A.sub(BC));
    }

    @Test
    void realSparseSubTestCase() {
        double[] bEntries;
        int[] bRowIndices;
        int[] bColIndices;
        Shape bShape;
        CooMatrixOld B;

        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new MatrixOld(aEntries);

        bEntries = new double[]{-0.99, 1, 14.2, 8.3};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = A.shape;
        B = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[]{1, 2+0.99, 3, 4-1, 5, 6-14.2, 7, 8, 9, 10-8.3, 11, 12};
        expShape = A.shape;
        exp = new MatrixOld(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new MatrixOld(aEntries);

        bEntries = new double[]{-0.99, 1, 14.2, 8.3};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = new Shape(4, 3);
        B = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }


    @Test
    void complexSparseSubTestCase() {
        CNumber[] bEntries;
        int[] bRowIndices;
        int[] bColIndices;
        Shape bShape;
        CooCMatrixOld B;

        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        A = new MatrixOld(aEntries);

        bEntries = new CNumber[]{new CNumber(-0.123, 1), new CNumber(0, 1),
                new CNumber(8, 3.3), new CNumber(100.23, -1000.2)};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = A.shape;
        B = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expEntriesC = new CNumber[]{
                new CNumber(1), new CNumber(2+0.123, -1), new CNumber(3),
                new CNumber(4, -1), new CNumber(5), new CNumber(6-8, -3.3),
                new CNumber(7), new CNumber(8), new CNumber(9),
                new CNumber(10-100.23, 1000.2), new CNumber(11), new CNumber(12)};
        expShape = A.shape;
        expC = new CMatrixOld(expShape, expEntriesC);

        assertEquals(expC, A.sub(B));

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        A = new MatrixOld(aEntries);

        bEntries = new CNumber[]{new CNumber(-0.123, 1), new CNumber(0, 1),
                new CNumber(8, 3.3), new CNumber(100.23, -1000.2)};
        bRowIndices = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{1, 0, 2, 0};
        bShape = new Shape(4, 3);
        B = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }
}
