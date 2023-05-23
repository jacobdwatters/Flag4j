package com.flag4j.sparse_vector;

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseVectorElemDivTests {

    SparseVector a;
    int size;

    @Test
    void denseElemDivTest() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new SparseVector(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        Vector b;
        double[] expValues;
        SparseVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new Vector(bValues);

        expValues = new double[]{1.34/1.223, 51.6/3.4, -0.00245/14.5};
        expIndices = new int[]{0, 2, 4};
        exp = new SparseVector(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 123, 25.2, 155};
        b = new Vector(bValues);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void denseComplexElemDivTest() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new SparseVector(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVector b;
        SparseCVector exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4),
                new CNumber(9924.515, 51.5)};
        b = new CVector(bValues);

        expValues = new CNumber[]{
                new CNumber(1.34).div(new CNumber(24.3, -0.013)),
                new CNumber(51.6).div(new CNumber(2.4)),
                new CNumber(-0.00245).div(new CNumber(1495, 13.4))
        };
        expIndices = new int[]{0, 2, 4};
        exp = new SparseCVector(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6)};
        b = new CVector(bValues);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.elemDiv(finalB));
    }
}
