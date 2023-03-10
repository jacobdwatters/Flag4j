/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.complex_vector;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CVectorEqualsTests {

    CNumber[] aEntries;
    CVector a;
    int sparseSize;
    int[] sparseIndices;

    @Test
    void realDenseTest() {
        double[] bEntries;
        Vector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, 0};
        b = new Vector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(-19345.612)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, -19345.612};
        b = new Vector(bEntries);
        assertEquals(a, b);

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(0)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3};
        b = new Vector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(0, 1.2)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, 0};
        b = new Vector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(1.334), new CNumber(0.645),
                new CNumber(1.3), new CNumber(-7234.5)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, -72};
        b = new Vector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 6 -----------------
        a = new CVector(2495, 1.45);
        b = new Vector(2495, 1.45);
        assertEquals(a, b);
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                new CNumber(1.3), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, 1.3};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertEquals(a, b);

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                new CNumber(), new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertEquals(a, b);


        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                new CNumber(), new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = 612345;
        sparseIndices = new int[]{1, 3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                new CNumber(), new CNumber(-99.1331, 1.23)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(0.1), new CNumber(8.245),
                new CNumber(), new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 6 -----------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(8.245),
                new CNumber(), new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 7 -----------------
        aEntries = new CNumber[]{new CNumber(), new CNumber(8.245),
                new CNumber(), new CNumber(-3.7)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);
    }


    @Test
    void complexDenseTest(){
        CNumber[] bEntries;
        CVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        b = new CVector(bEntries);
        assertEquals(a, b);

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(8.124, 9.4), new CNumber(1.55),
                new CNumber(0, -85.215), new CNumber(0.000013, 14.5),
                new CNumber(1.335676, -89345)};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.124, 9.4), new CNumber(1.55),
                new CNumber(0, -85.215), new CNumber(0.000013, 14.5),
                new CNumber(1.335676, -89345)};
        b = new CVector(bEntries);
        assertEquals(a, b);

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(8.124, 9.4), new CNumber(1.55),
                new CNumber(0, -85.215), new CNumber(0.000013, 14.5),
                new CNumber(1.335676, -89345)};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.124, 9.4), new CNumber(1.55),
                new CNumber(0, -85.215), new CNumber(0.000013, 14.5)};
        b = new CVector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 4 -----------------
        a = new CVector(45, -92341.566);
        b = new CVector(45, -92341.566);
        assertEquals(a, b);

        // ----------------- Sub-case 5 -----------------
        a = new CVector(45, -92341.566);
        b = new CVector(41, -92341.566);
        assertNotEquals(a, b);

        // ----------------- Sub-case 6 -----------------
        a = new CVector(45, new CNumber("92.1465+879234.9999324i"));
        b = new CVector(45, new CNumber("92.1465+879234.9999324i"));
        assertEquals(a, b);

        // ----------------- Sub-case 7 -----------------
        a = new CVector(45, new CNumber("92.1465+879234.9999324i"));
        b = new CVector(41, new CNumber("92.1465+879234.9999324i"));
        assertNotEquals(a, b);

        // ----------------- Sub-case 8 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.2341), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        b = new CVector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 9 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber(1.2)};
        b = new CVector(bEntries);
        assertNotEquals(a, b);
    }


    @Test
    void complexSparseTest() {
        CNumber[] bEntries;
        SparseCVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        assertEquals(a, b);

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 13.65), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(12.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(12.3, -0.000023465)};
        sparseSize = 100234;
        sparseIndices = new int[]{1, 2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        assertNotEquals(a, b);
    }


    @Test
    void objectTest() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        String bString = "Hello World!";
        assertFalse(a.equals(bString));

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        Double num = 123.4;
        assertFalse(a.equals(num));

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        a = new CVector(aEntries);
        CNumber[] arr = {new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), new CNumber()};
        assertFalse(a.equals(num));
    }
}
