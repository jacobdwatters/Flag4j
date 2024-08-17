package org.flag4j.vector;


import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class VectorConstructorTests {

    int expSize;
    double fillValue;
    Shape expShape, shape;
    double[] expEntries;
    int[] entriesI;
    VectorOld a, b;

    @Test
    void sizeTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];

        a = new VectorOld(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];

        a = new VectorOld(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new VectorOld(expSize));
    }

    @Test
    void sizeFillTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -10.23423;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new VectorOld(expSize, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -10.23423;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new VectorOld(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValue = -10.23423;
        assertThrows(IllegalArgumentException.class, () -> new VectorOld(expSize, fillValue));
    }


    @Test
    void entriesTestCase() {
        // ----------- Sub-case 1 ------------
        expEntries = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new VectorOld(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expEntries = new double[]{-0.234974};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new VectorOld(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void entriesITestCase() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new VectorOld(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new VectorOld(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void copyTestCase() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new VectorOld(entriesI);
        a = new VectorOld(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new VectorOld(entriesI);
        a = new VectorOld(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new VectorOld(entriesI);
        a = new VectorOld(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new VectorOld(entriesI);
        a = new VectorOld(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void shapeTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];

        a = new VectorOld(expShape);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];

        a = new VectorOld(expShape);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expShape = new Shape(1, 4);
        assertThrows(IllegalArgumentException.class, () -> new VectorOld(expShape));
    }

    @Test
    void shapeFillTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -10.23423;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new VectorOld(expShape, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -10.23423;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new VectorOld(expShape,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        fillValue = -10.23423;
        expShape = new Shape(1, 14, 6);
        assertThrows(IllegalArgumentException.class, () -> new VectorOld(expShape, fillValue));
    }
}
