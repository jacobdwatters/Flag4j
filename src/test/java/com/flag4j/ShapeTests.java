package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ShapeTests {
    int expRank, expValue;
    int[] expDims1, expDims2;
    Shape shape1;
    Shape shape2;


    @Test
    void dimsConstructorTest() {
        // ----------- Sub-case 1 -----------
        expRank = 3;
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);

        assertEquals(expRank, shape1.getRank());
        assertArrayEquals(expDims1, shape1.getDims());

        // ----------- Sub-case 2 -----------
        expRank = 1;
        expDims1 = new int[]{8};
        shape1 = new Shape(expDims1);

        assertEquals(expRank, shape1.getRank());
        assertArrayEquals(expDims1, shape1.getDims());

        // ----------- Sub-case 3 -----------
        expRank = 13;
        expDims1 = new int[]{8, 1, 2, 3, 4, 1, 10, 19, 304, 11, 8, 3, 90};
        shape1 = new Shape(expDims1);

        // ----------- Sub-case 4 -----------
        expDims1 = new int[]{1, 3, -1, 0};
        assertThrows(IllegalArgumentException.class, () -> new Shape(expDims1));
    }


    @Test
    void getTest() {
        // ----------- Sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);

        for(int i = 0; i< expDims1.length; i++) {
            assertEquals(expDims1[i], shape1.get(i));
        }

        // ----------- Sub-case 3 -----------
        expDims1 = new int[]{4, 903, 11, 45};
        shape1 = new Shape(expDims1);

        for(int i = 0; i< expDims1.length; i++) {
            assertEquals(expDims1[i], shape1.get(i));
        }
    }


    @Test
    void totalEntriesTest() {
        // ----------- Sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);
        assertEquals(1*5*18, shape1.totalEntries().intValue());

        // ----------- Sub-case 2 -----------
        expDims1 = new int[]{4, 903, 11, 45};
        shape1 = new Shape(expDims1);
        assertEquals(4*903*11*45, shape1.totalEntries().intValue());

        // ----------- Sub-case 3 -----------
        expDims1 = new int[]{};
        shape1 = new Shape(expDims1);
        assertEquals(0, shape1.totalEntries().intValue());
    }


    @Test
    void equalsTest() {
        // ----------- Sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        expDims2 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertTrue(shape1.equals(shape2));

        // ----------- Sub-case 2 -----------
        expDims1 = new int[]{1, 5, 18};
        expDims2 = new int[]{1, 5, 18, 0};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertFalse(shape1.equals(shape2));

        // ----------- Sub-case 3 -----------
        expDims1 = new int[]{1};
        expDims2 = new int[]{4};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertFalse(shape1.equals(shape2));

        // ----------- Sub-case 4 -----------
        expDims1 = new int[]{};
        expDims2 = new int[]{};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertTrue(shape1.equals(shape2));

        // ----------- Sub-case 5 -----------
        expDims1 = new int[]{};
        shape1 = new Shape(expDims1);

        assertFalse(shape1.equals(Double.valueOf(1)));
    }
}
