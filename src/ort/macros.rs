/// Convert multiple tensors to ORT SessionInputs.
///
/// This macro provides a convenient way to convert `X` or `XView` tensors to ORT input format.
/// It supports both positional and named input styles, and allows **mixing** different input types.
///
/// **Note**: This macro does NOT perform dtype alignment. The tensors must already have
/// the correct dtype expected by the model. For automatic dtype alignment, use `&[X]` input.
///
/// # Input Types (can be mixed freely)
/// - `X<A>` (owned) → moves data into ORT tensor
/// - `&X<A>` (borrowed) → clones data into ORT tensor
/// - `XView<'a, A>` (view) → **zero-copy** reference to original data
///
/// # Examples
/// ```ignore
/// // Owned inputs
/// let inputs = inputs![x1, x2]?;
///
/// // Borrowed inputs
/// let inputs = inputs![&x1, &x2]?;
///
/// // Zero-copy views
/// let inputs = inputs![x1.view(), x2.view()]?;
///
/// // Mixed inputs (owned + borrowed + view)
/// let inputs = inputs![x1, &x2, x3.view()]?;
///
/// // By name
/// let inputs = inputs!["images" => x1, "masks" => &x2]?;
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! inputs {
    // Positional inputs
    ($($t:expr),+ $(,)?) => {{
        (|| -> ::anyhow::Result<_> {
            Ok([ $(
                ::ort::session::SessionInputValue::try_from($t)?,
            )+ ])
        })()
    }};
    // Named inputs
    ($($name:expr => $t:expr),+ $(,)?) => {{
        (|| -> ::anyhow::Result<_> {
            Ok(vec![ $(
                (
                    ::std::borrow::Cow::<str>::from($name),
                    ::ort::session::SessionInputValue::try_from($t)?
                ),
            )+ ])
        })()
    }};
}
