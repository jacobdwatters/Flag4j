package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixAddTests {

    double[][] aEntries, bEntries;
    CNumber[][] bCEntries;

    Matrix A, B;
    CMatrix BC;
    double b;
    CNumber bC;
    Matrix sum, exp;
    CMatrix sumC;
    Shape expShape;
    double[] expEntries;
    CNumber[] expEntriesC;

    @Test
    void matrixMatrixTest() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1.343, 6.7, -88.4}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        expShape = A.shape.clone();
        expEntries = new double[]{1+0.333, 2+56.4, 3+13.4, 4-1.44, 5+5, 6+85.1, 7+1.343, 8+6.7, 9-88.4};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(B);

        assertEquals(exp, sum);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        expShape = A.shape.clone();
        expEntries = new double[]{1+0.333, 2+56.4, 3+13.4, 4 - 1.44, 5+5, 6+85.1};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(B);

        assertEquals(exp, sum);

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}, {1, 2, 3}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        assertThrows(IllegalArgumentException.class, ()->A.add(B));

        // --------------- Sub-case 4 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bEntries = new double[][]{{0.333, 56.4, 13.4}, {-1.44, 5, 85.1}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        assertThrows(IllegalArgumentException.class, ()->A.add(B));
    }


    @Test
    void matrixDoubleTest() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        b = 2.133;
        A = new Matrix(aEntries);
        expShape = A.shape.clone();
        expEntries = new double[]{1+2.133, 2+2.133, 3+2.133, 4+2.133, 5+2.133, 6+2.133, 7+2.133, 8+2.133, 9+2.133};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(b);

        assertEquals(exp, sum);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        b = 2.133;
        A = new Matrix(aEntries);
        expShape = A.shape.clone();
        expEntries = new double[]{1+2.133, 2+2.133, 3+2.133, 4+2.133, 5+2.133, 6+2.133};
        exp = new Matrix(expShape, expEntries);

        sum = A.add(b);

        assertEquals(exp, sum);
    }


    @Test
    void matrixCNumberTest() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bC = new CNumber(33.444, -9.3545);
        A = new Matrix(aEntries);
        expShape = A.shape.clone();
        expEntriesC = new CNumber[]{
                new CNumber(1).add(bC), new CNumber(2).add(bC), new CNumber(3).add(bC),
                new CNumber(4).add(bC), new CNumber(5).add(bC), new CNumber(6).add(bC),
                new CNumber(7).add(bC), new CNumber(8).add(bC), new CNumber(9).add(bC)
        };

        sumC = A.add(bC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);

        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bC = new CNumber(33.444, -9.3545);
        A = new Matrix(aEntries);
        expShape = A.shape.clone();
        expEntriesC = new CNumber[]{
                new CNumber(1).add(bC), new CNumber(2).add(bC), new CNumber(3).add(bC),
                new CNumber(4).add(bC), new CNumber(5).add(bC), new CNumber(6).add(bC)
        };

        sumC = A.add(bC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);
    }


    @Test
    void matrixCMatrixTest() {
        // --------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)},
                {new CNumber(1.23), new CNumber(8, 3), new CNumber(9, -0.000000001)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);
        expShape = A.shape.clone();
        expEntriesC = new CNumber[]{
                new CNumber(1.23+1, -344.5), new CNumber(2.33+2, 5.6), new CNumber(3.13+3, -34),
                new CNumber(4, 66.45), new CNumber(33.1334+5, 5513.5), new CNumber(99.3+6),
                new CNumber(1.23+7), new CNumber(8+8, 3), new CNumber(9+9, -0.000000001)
        };

        sumC = A.add(BC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);


        // --------------- Sub-case 2 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);
        expShape = A.shape.clone();
        expEntriesC = new CNumber[]{
                new CNumber(1.23+1, -344.5), new CNumber(2.33+2, 5.6), new CNumber(3.13+3, -34),
                new CNumber(4, 66.45), new CNumber(33.1334+5, 5513.5), new CNumber(99.3+6)
        };

        sumC = A.add(BC);

        assertArrayEquals(expEntriesC, sumC.entries);
        assertEquals(expShape, sumC.shape);

        // --------------- Sub-case 3 ---------------
        aEntries = new double[][]{{1, 2}, {4, 5}};
        bCEntries = new CNumber[][]{
                {new CNumber(1.23, -344.5), new CNumber(2.33, 5.6), new CNumber(3.13, -34)},
                {new CNumber(0, 66.45), new CNumber(33.1334, 5513.5), new CNumber(99.3)}
        };
        A = new Matrix(aEntries);
        BC = new CMatrix(bCEntries);

        assertThrows(IllegalArgumentException.class, ()->A.add(BC));
    }
}
