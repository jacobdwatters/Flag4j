package com.flag4j;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorPropertiesTests {

    Shape shape;
    double[] entries;
    Tensor A;


    @Test
    void isZerosTest() {
        // ----------------------- Sub-case 1 -----------------------
        entries = new double[1024*4*9];
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isZeros());

        // ----------------------- Sub-case 2 -----------------------
        entries = new double[1024*4*9];
        entries[1345] = 1.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isZeros());

        // ----------------------- Sub-case 3 -----------------------
        entries = new double[]{0, 0.0, -0.0};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertTrue(A.isZeros());

        // ----------------------- Sub-case 4 -----------------------
        entries = new double[]{0, 0.0, -0.0, 1.34};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertFalse(A.isZeros());
    }


    @Test
    void isOnesTest() {
        // ----------------------- Sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isOnes());

        // ----------------------- Sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1);
        entries[1345] = -131.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isOnes());

        // ----------------------- Sub-case 3 -----------------------
        entries = new double[]{1, 1.0, 1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertTrue(A.isOnes());

        // ----------------------- Sub-case 4 -----------------------
        entries = new double[]{1, 1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isOnes());
    }


    @Test
    void isPosTest() {
        // ----------------------- Sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 24.0);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isPos());

        // ----------------------- Sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1515.11331);
        entries[1345] = -131.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isPos());

        // ----------------------- Sub-case 3 -----------------------
        entries = new double[]{144, 1, 1.31415, 512.234345};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertTrue(A.isPos());

        // ----------------------- Sub-case 4 -----------------------
        entries = new double[]{1, 1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isPos());
    }


    @Test
    void isNegTest() {
        // ----------------------- Sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, -24.0);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isNeg());

        // ----------------------- Sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, -1515.11331);
        entries[1345] = 1.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isNeg());

        // ----------------------- Sub-case 3 -----------------------
        entries = new double[]{-144, -1, -1.31415, -512.234345};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertTrue(A.isNeg());

        // ----------------------- Sub-case 4 -----------------------
        entries = new double[]{1, -1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isNeg());
    }
}
