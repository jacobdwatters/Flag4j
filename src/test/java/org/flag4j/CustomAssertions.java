package org.flag4j;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;

import static org.junit.jupiter.api.Assertions.*;

public final class CustomAssertions {


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(Matrix exp, Matrix act) {
        assertEquals(exp.shape, act.shape);
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                assertEquals(exp.data[i], act.data[i], 0);
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
        for(int i = 0; i<exp.data.length; i++) {
            if(exp.data[i].isNaN()) {
                if(Double.isNaN(((Complex128) exp.data[i]).re)) {
                    assertTrue(Double.isNaN(((Complex128) act.data[i]).re));
                } else {
                    assertEquals(((Complex128) exp.data[i]).re, ((Complex128) act.data[i]).re, 0);
                }

                if(Double.isNaN(((Complex128) exp.data[i]).im)) {
                    assertTrue(Double.isNaN(((Complex128) act.data[i]).im));
                } else {
                    assertEquals(((Complex128) exp.data[i]).im, ((Complex128) act.data[i]).im, 0);
                }
            } else {
                assertEquals(exp.data[i], act.data[i]);
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
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                assertEquals(exp.data[i], act.data[i], 0);
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
        for(int i = 0; i<exp.data.length; i++) {
            if(Double.isNaN(exp.data[i])) {
                assertTrue(Double.isNaN(act.data[i]));
            } else {
                assertEquals(exp.data[i], act.data[i], 0);
            }
        }
    }
}
