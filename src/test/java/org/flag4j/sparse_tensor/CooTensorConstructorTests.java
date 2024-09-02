package org.flag4j.sparse_tensor;


import org.flag4j.arrays_old.sparse.CooTensorOld;
import org.flag4j.arrays.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooTensorConstructorTests {
    double[] expNonZero;
    int[] expNonZeroI;
    int[][] expIndices;
    Shape expShape;
    CooTensorOld A, B;

    @Test
    void shapeTestCase() {
        // --------------- Sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[0];
        expIndices = new int[0][0];

        A = new CooTensorOld(expShape);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZero = new double[0];
        expIndices = new int[0][0];

        A = new CooTensorOld(expShape);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);
    }


    @Test
    void shapeEntriesIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[]{1, 223.1333, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};

        A = new CooTensorOld(expShape, expNonZero, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZero = new double[]{1, 223.1333};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        A = new CooTensorOld(expShape, expNonZero, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);


        // --------------- Sub-case 3 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333, 50, 133, 1335.34};
        expIndices = new int[][]{{95, 1}, {96, 0}, {97, 1}, {98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZero, expIndices));

        // --------------- Sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333, 50, 133};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZero, expIndices));

        // --------------- Sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZero = new double[]{1, 223.1333};
        expIndices = new int[][]{{98, 0, 1}, {99, 1, 8}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZero, expIndices));
    }


    @Test
    void shapeEntriesIntIndicesTestCase() {
        // --------------- Sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expNonZeroI = new int[]{1, 223, -19, 2};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};

        A = new CooTensorOld(expShape, expNonZeroI, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(100, 2);
        expNonZeroI = new int[]{1, 223};
        expNonZero = Arrays.stream(expNonZeroI).asDoubleStream().toArray();
        expIndices = new int[][]{{98, 0}, {99, 1}};

        A = new CooTensorOld(expShape, expNonZeroI, expIndices);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);

        // --------------- Sub-case 3 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223, 50, 133, 1335};
        expIndices = new int[][]{{95, 1}, {96, 0}, {97, 1}, {98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZeroI, expIndices));

        // --------------- Sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223, 50, 133};
        expIndices = new int[][]{{98, 0}, {99, 1}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZeroI, expIndices));

        // --------------- Sub-case 4 ---------------
        expShape = new Shape(2, 2);
        expNonZeroI = new int[]{1, 223};
        expIndices = new int[][]{{98, 0, 1}, {99, 1, 8}};

        assertThrows(IllegalArgumentException.class, () -> new CooTensorOld(expShape, expNonZeroI, expIndices));
    }


    @Test
    void copyTestCase() {
        expShape = new Shape(3, 4, 5, 1);
        expNonZero = new double[]{1, 223.1333, -0.991233, 100.1234};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};

        B = new CooTensorOld(expShape, expNonZero, expIndices);
        A = new CooTensorOld(B);

        assertEquals(expShape, A.getShape());
        assertArrayEquals(expNonZero, A.entries);
        assertArrayEquals(expIndices, A.indices);
    }
}
