package org.flag4j.sparse_complex_tensor;

import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCTensorConstructorTests {

    Shape expShape;
    CNumber[] expNonZero;
    double[] expNonZeroD;
    int[] expNonZeroI;
    int[][] expIndices;
    CooCTensor A, B;


    @Test
    void shapeTestCase() {
        // ------------ Sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new CNumber[0];
        expIndices = new int[0][0];
        A = new CooCTensor(expShape);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }

        // ------------ Sub-case 2 ------------
        expShape = new Shape(12);
        expNonZero = new CNumber[0];
        expIndices = new int[0][0];
        A = new CooCTensor(expShape);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // ------------ Sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new CNumber[]{new CNumber(1, -0.92342), new CNumber(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        A = new CooCTensor(expShape, expNonZero, expIndices);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }

        // ------------ Sub-case 2 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new CNumber[]{new CNumber(1, -0.92342), new CNumber(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}, {0, 1, 22, 3, 0, 10}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));

        // ------------ Sub-case 3 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new CNumber[]{new CNumber(1, -0.92342), new CNumber(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0}, {0, 1, 22, 2, 0}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));

        // ------------ Sub-case 4 ------------
        expShape = new Shape(2);
        expNonZero = new CNumber[]{new CNumber(1, -0.92342), new CNumber(-100, 123.44), new CNumber(0, 1)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZero, expIndices));
    }


    @Test
    void shapeEntriesIndicesDTestCase() {
        // ------------ Sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new CNumber[expNonZeroD.length];
        ArrayUtils.copy2CNumber(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}, {0, 1, 24, 2, 0, 10},
                {0, 1, 26, 2, 0, 10}, {0, 1, 28, 3, 0, 10}};
        A = new CooCTensor(expShape, expNonZeroD, expIndices);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }

        // ------------ Sub-case 2 ------------
        expShape = new Shape(5, 3);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new CNumber[expNonZeroD.length];
        ArrayUtils.copy2CNumber(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0}, {1, 1}, {2, 2}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));

        // ------------ Sub-case 3 ------------
        expShape = new Shape(5, 3);
        expNonZeroD = new double[]{1, 2};
        expNonZero = new CNumber[expNonZeroD.length];
        ArrayUtils.copy2CNumber(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0}, {0, 1, 22, 2, 0}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));

        // ------------ Sub-case 4 ------------
        expShape = new Shape(2);
        expNonZeroD = new double[]{1, 2, 3, 4.023423, -9233.2};
        expNonZero = new CNumber[expNonZeroD.length];
        ArrayUtils.copy2CNumber(expNonZeroD, expNonZero);
        expIndices = new int[][]{{0}, {1}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroD, expIndices));
    }


    @Test
    void shapeEntriesIndicesITestCase() {
        // ------------ Sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZeroI = new int[]{1, 2, 3, 400, -9233};
        expNonZero = new CNumber[expNonZeroI.length];
        ArrayUtils.copy2CNumber(expNonZeroI, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}, {0, 1, 24, 2, 0, 10},
                {0, 1, 26, 2, 0, 10}, {0, 1, 28, 3, 0, 10}};
        A = new CooCTensor(expShape, expNonZeroI, expIndices);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }

        // ------------ Sub-case 2 ------------
        expShape = new Shape(5, 3);
        expNonZeroI = new int[]{1, 2, 3, 4, -9233};
        expNonZero = new CNumber[expNonZeroI.length];
        ArrayUtils.copy2CNumber(expNonZeroI, expNonZero);
        expIndices = new int[][]{{0, 0}, {1, 1}, {2, 2}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroI, expIndices));

        // ------------ Sub-case 3 ------------
        expShape = new Shape(5, 3);
        expNonZeroI = new int[]{1, 2};
        expNonZero = new CNumber[expNonZeroI.length];
        ArrayUtils.copy2CNumber(expNonZeroI, expNonZero);
        expIndices = new int[][]{{0, 0, 10, 1, 0}, {0, 1, 22, 2, 0}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroI, expIndices));

        // ------------ Sub-case 4 ------------
        expShape = new Shape(2);
        expNonZeroI = new int[]{1, 2, 3, 4, -9233};
        expNonZero = new CNumber[expNonZeroI.length];
        ArrayUtils.copy2CNumber(expNonZeroI, expNonZero);
        expIndices = new int[][]{{0}, {1}};
        assertThrows(IllegalArgumentException.class, () -> new CooCTensor(expShape, expNonZeroI, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------ Sub-case 1 ------------
        expShape = new Shape(1, 2, 31, 4, 1, 11);
        expNonZero = new CNumber[]{new CNumber(1, -0.92342), new CNumber(-100, 123.44)};
        expIndices = new int[][]{{0, 0, 10, 1, 0, 9}, {0, 1, 22, 2, 0, 10}};
        B = new CooCTensor(expShape, expNonZero, expIndices);
        A = new CooCTensor(B);

        Assertions.assertEquals(expShape, A.getShape());
        Assertions.assertArrayEquals(expIndices, A.indices);
        Assertions.assertEquals(expNonZero.length, A.entries.length);
        for(int i=0; i<expNonZero.length; i++) {
            Assertions.assertEquals(expNonZero[i], A.entries[i]);
        }
    }
}
