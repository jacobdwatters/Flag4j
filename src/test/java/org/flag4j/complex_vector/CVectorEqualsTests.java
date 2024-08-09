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

package org.flag4j.complex_vector;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CVectorEqualsTests {

    CNumber[] aEntries;
    CVector a;
    int sparseSize;
    int[] sparseIndices;

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, 0};
        b = new Vector(bEntries);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(-19345.612)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, -19345.612};
        b = new Vector(bEntries);
        assertTrue(a.tensorEquals(b));

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(0)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3};
        b = new Vector(bEntries);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(1), new CNumber(0),
                new CNumber(1.3), new CNumber(0, 1.2)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, 0};
        b = new Vector(bEntries);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(1.334), new CNumber(0.645),
                new CNumber(1.3), new CNumber(-7234.5)};
        a = new CVector(aEntries);
        bEntries = new double[]{1, 0, 1.3, -72};
        b = new Vector(bEntries);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 6 -----------------
        a = new CVector(2495, 1.45);
        b = new Vector(2495, 1.45);
        assertTrue(a.tensorEquals(b));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                new CNumber(1.3), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, 1.3};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertTrue(a.tensorEquals(b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                CNumber.ZERO, new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertTrue(a.tensorEquals(b));


        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                CNumber.ZERO, new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = 612345;
        sparseIndices = new int[]{1, 3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245),
                CNumber.ZERO, new CNumber(-99.1331, 1.23)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(0.1), new CNumber(8.245),
                CNumber.ZERO, new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 6 -----------------
        aEntries = new CNumber[]{CNumber.ZERO, new CNumber(8.245),
                CNumber.ZERO, new CNumber(-99.1331)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 7 -----------------
        aEntries = new CNumber[]{CNumber.ZERO, new CNumber(8.245),
                CNumber.ZERO, new CNumber(-3.7)};
        a = new CVector(aEntries);
        bEntries = new double[]{8.245, -99.1331};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));
    }


    @Test
    void complexDenseTestCase(){
        CNumber[] bEntries;
        CVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), CNumber.ZERO};
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
                new CNumber(1.3), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.2341), new CNumber(0, 8.245),
                new CNumber(1.3), CNumber.ZERO};
        b = new CVector(bEntries);
        assertNotEquals(a, b);

        // ----------------- Sub-case 9 -----------------
        aEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(1, -9.234), new CNumber(0, 8.245),
                new CNumber(1.3), new CNumber(1.2)};
        b = new CVector(bEntries);
        assertNotEquals(a, b);
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        assertTrue(a.tensorEquals(b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 3};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 13.65), new CNumber(1.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 4 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(12.3, -0.000023465)};
        sparseSize = aEntries.length;
        sparseIndices = new int[]{1, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));

        // ----------------- Sub-case 5 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        bEntries = new CNumber[]{new CNumber(8.245, 9.2165), new CNumber(12.3, -0.000023465)};
        sparseSize = 100234;
        sparseIndices = new int[]{1, 2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        assertFalse(a.tensorEquals(b));
    }


    @Test
    void objectTestCase() {
        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        String bString = "Hello World!";
        assertNotEquals(a, bString);

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        Double num = 123.4;
        assertNotEquals(a, num);

        // ----------------- Sub-case 3 -----------------
        aEntries = new CNumber[]{new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        a = new CVector(aEntries);
        CNumber[] arr = {new CNumber(0), new CNumber(8.245, 9.2165),
                new CNumber(1.3, -0.000023465), CNumber.ZERO};
        assertNotEquals(a, num);
    }
}
