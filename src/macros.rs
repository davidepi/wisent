#[macro_export]
macro_rules! fxhashset {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(fxhashset!(@single $rest)),*]));

    ($($key:expr,)+) => { fxhashset!($($key),+) };
    ($($key:expr),*) => {
        {
            let _cap = fxhashset!(@count $($key),*);
            let _h = ::std::hash::BuildHasherDefault::<rustc_hash::FxHasher>::default();
            let mut _set = ::std::collections::HashSet::with_capacity_and_hasher(_cap, _h);
            $(
                let _ = _set.insert($key);
            )*
            _set
        }
    };
}
