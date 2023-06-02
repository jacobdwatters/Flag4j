package com.flag4j.vector;


import com.flag4j.Shape;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class VectorConstructorTests {

    int expSize;
    double fillValue;
    Shape expShape, shape;
    double[] expEntries;
    int[] entriesI;
    Vector a, b;

    @Test
    void sizeTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];

        a = new Vector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];

        a = new Vector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new Vector(expSize));
    }

    @Test
    void sizeFillTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValue = -10.23423;
        assertThrows(IllegalArgumentException.class, () -> new Vector(expSize, fillValue));
    }


    @Test
    void entriesTest() {
        // ----------- Sub-case 1 ------------
        expEntries = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new Vector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expEntries = new double[]{-0.234974};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new Vector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void entriesITest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        a = new Vector(entriesI);

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

        a = new Vector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void copyTest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI);
        a = new Vector(b);

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

        b = new Vector(entriesI);
        a = new Vector(b);

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

        b = new Vector(entriesI);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);;
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new double[expSize];
        for(int i=0; i<entriesI.length; i++) {
            expEntries[i] = entriesI[i];
        }

        b = new Vector(entriesI);
        a = new Vector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }


    @Test
    void shapeTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];

        a = new Vector(expShape);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];

        a = new Vector(expShape);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        expShape = new Shape(1, 4);
        assertThrows(IllegalArgumentException.class, () -> new Vector(expShape));
    }

    @Test
    void shapeFillTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expShape, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = -10.23423;
        expShape = new Shape(new int[]{expSize});
        expEntries = new double[expSize];
        Arrays.fill(expEntries, fillValue);

        a = new Vector(expShape,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ----------- Sub-case 3 ------------
        fillValue = -10.23423;
        expShape = new Shape(1, 14, 6);
        assertThrows(IllegalArgumentException.class, () -> new Vector(expShape, fillValue));
    }
}
