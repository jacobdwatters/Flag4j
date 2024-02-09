package com.flag4j;

import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import com.flag4j.sparse.CooMatrix;
import com.flag4j.sparse.CooVector;
import junit.framework.Assert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public final class CustomAssertions extends Assert {


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(Matrix exp, Matrix act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(CMatrix exp, CMatrix act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(exp.entries[i].isNaN()) {
                if(Double.isNaN(exp.entries[i].re)) {
                    assertTrue(Double.isNaN(act.entries[i].re));
                } else {
                    assertEquals(exp.entries[i].re, act.entries[i].re);
                }

                if(Double.isNaN(exp.entries[i].im)) {
                    assertTrue(Double.isNaN(act.entries[i].im));
                } else {
                    assertEquals(exp.entries[i].im, act.entries[i].im);
                }
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }


    /**
     * Checks if sparse matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(CooMatrix exp, CooMatrix act) {
        assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.rowIndices, act.colIndices);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }


    /**
     * Checks if sparse vectors are equal counting NaN equal to NaN.
     * @param exp Expected vector.
     * @param act Actual vector.
     */
    public static void assertEqualsNaN(CooVector exp, CooVector act) {
        assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.indices, act.indices);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }
}
