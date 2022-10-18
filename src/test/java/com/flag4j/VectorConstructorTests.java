package com.flag4j;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorConstructorTests {
    private static Vector a;
    private static Vector b;
    private static double[] expEntries;
    private static int[] expEntriesInt;
    private static VectorOrientations expOrientation;
    private static int expSize;

    @Test
    void defaultConstructorTestCase() {
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[0];
        expSize = 0;
        a = new Vector();

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void orientConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[0];
        expSize = 0;
        a = new Vector(VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[0];
        expSize = 0;
        a = new Vector(VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void sizeConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[10];
        expSize = 10;
        a = new Vector(10);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[0];
        expSize = 0;
        a = new Vector(0);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 3 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-1));

        // ------------ Sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-5));
    }


    @Test
    void sizeOrientConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[10];
        expSize = 10;
        a = new Vector(10, VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[14];
        expSize = 14;
        a = new Vector(14, VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 3 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-1, VectorOrientations.COLUMN_VECTOR));

        // ------------ Sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-5, VectorOrientations.ROW_VECTOR));
    }


    @Test
    void sizeValueConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[10];
        expSize = 10;
        a = new Vector(10, 90.13);
        Arrays.fill(expEntries, 90.13);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[14];
        expSize = 14;
        a = new Vector(14,-1033);
        Arrays.fill(expEntries, -1033);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 3 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-1, 93));

        // ------------ Sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () -> new Vector(-5, 15));
    }


    @Test
    void sizeValueOrientConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[10];
        expSize = 10;
        a = new Vector(10, 90.13, VectorOrientations.COLUMN_VECTOR);
        Arrays.fill(expEntries, 90.13);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[14];
        expSize = 14;
        a = new Vector(14,-1033, VectorOrientations.ROW_VECTOR);
        Arrays.fill(expEntries, -1033);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 3 ------------
        assertThrows(IllegalArgumentException.class, () ->
                new Vector(-1, 93, VectorOrientations.ROW_VECTOR));

        // ------------ Sub-case 4 ------------
        assertThrows(IllegalArgumentException.class, () ->
                new Vector(-5, 15, VectorOrientations.COLUMN_VECTOR));
    }


    @Test
    void entriesConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[]{1, 3, -1, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void entriesOrientConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[]{1, 3, -1, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[]{1, 3, -1, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void entriesIntConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntriesInt = new int[]{1, 3, -1};
        expEntries = new double[]{1, 3, -1};
        expSize = expEntries.length;
        a = new Vector(expEntriesInt);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void entriesIntOrientConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntriesInt = new int[]{1, 3, -1};
        expEntries = new double[]{1, 3, -1};
        expSize = expEntries.length;
        a = new Vector(expEntriesInt, VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);

        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[]{1, 3, -1};
        expEntries = new double[]{1, 3, -1};
        expSize = expEntries.length;
        a = new Vector(expEntriesInt, VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, a.entries);
        assertEquals(expSize, a.m);
        assertEquals(expOrientation, a.orientation);
    }


    @Test
    void vecConstructorTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries);
        b = new Vector(a);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.ROW_VECTOR);
        b = new Vector(a);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);
    }


    @Test
    void vecOrientTestCase() {
        // ------------ Sub-case 1 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries);
        b = new Vector(a, VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);


        // ------------ Sub-case 2 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.COLUMN_VECTOR);
        b = new Vector(a, VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);

        // ------------ Sub-case 3 ------------
        expOrientation = VectorOrientations.COLUMN_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.ROW_VECTOR);
        b = new Vector(a, VectorOrientations.COLUMN_VECTOR);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);


        // ------------ Sub-case 4 ------------
        expOrientation = VectorOrientations.ROW_VECTOR;
        expEntries = new double[]{4.3, -1, 201.123, 1e-4, Double.NEGATIVE_INFINITY};
        expSize = expEntries.length;
        a = new Vector(expEntries, VectorOrientations.ROW_VECTOR);
        b = new Vector(a, VectorOrientations.ROW_VECTOR);

        assertArrayEquals(expEntries, b.entries);
        assertEquals(expSize, b.m);
        assertEquals(expOrientation, b.orientation);
    }
}
