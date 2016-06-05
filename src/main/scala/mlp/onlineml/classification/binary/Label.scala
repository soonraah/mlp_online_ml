package mlp.onlineml.classification.binary

case class Label(clazz: Boolean) {
  def value: Int = if (clazz) 1 else -1
  override def toString(): String = value.toString
}
